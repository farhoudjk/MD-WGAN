/*
 * mdwgan-comm.cc - Communication-cost measurement for the MD-WGAN framework.
 *
 * Measures what a federated synchronisation round actually costs over an IEEE
 * 802.11p / DSRC vehicular channel: the RSU (generator) broadcasts a synthetic
 * batch to its selected discriminators, each replies with critic feedback, and
 * the round closes when the last reply arrives.
 *
 * This is deliberately decoupled from model training.  The learning dynamics
 * are measured by the Python harness; here we vary only the network-side
 * parameters (vehicle count, selected subset size, payload, mobility) so the
 * reported latency and overhead come from the simulated stack rather than from
 * the byte size of files written to local disk.
 *
 * Output: one CSV row per (round, config) to --out.
 *
 * Example:
 *   ./ns3 run "mdwgan-comm --nVehicles=20 --kSelected=3 --rounds=60"
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <map>
#include <vector>
#include <deque>
#include <algorithm>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MdWganComm");

namespace {

// Tags a reply with the round it answers, so the generator attributes each
// arrival to the correct synchronisation cycle rather than to whichever round
// happens to still be outstanding.
class ReplyTag : public Tag {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("ReplyTag")
                            .SetParent<Tag>()
                            .AddConstructor<ReplyTag>();
    return tid;
  }
  TypeId GetInstanceTypeId() const override { return GetTypeId(); }
  uint32_t GetSerializedSize() const override { return 4; }
  void Serialize(TagBuffer i) const override { i.WriteU32(round); }
  void Deserialize(TagBuffer i) override { round = i.ReadU32(); }
  void Print(std::ostream &os) const override { os << "round=" << round; }
  uint32_t round = 0;
};

struct RoundStat {
  double startTime = 0.0;
  double lastReplyTime = 0.0;
  uint32_t repliesExpected = 0;
  uint32_t repliesReceived = 0;
  uint64_t bytesDown = 0;
  uint64_t bytesUp = 0;
};

class FederatedRound {
public:
  FederatedRound(uint32_t kSelected, uint32_t payloadBytes, uint32_t rounds,
                 double interval, std::string outPath, uint32_t seed)
      : m_k(kSelected), m_payload(payloadBytes), m_rounds(rounds),
        m_interval(interval), m_out(outPath), m_seed(seed) {}

  void Setup(NodeContainer rsus, NodeContainer vehicles,
             Ipv4InterfaceContainer rsuIf, Ipv4InterfaceContainer vehIf) {
    m_rsus = rsus;
    m_vehicles = vehicles;
    m_rsuIf = rsuIf;
    m_vehIf = vehIf;
    m_rng = CreateObject<UniformRandomVariable>();
    m_rng->SetStream(m_seed);

    // One receive socket per vehicle (discriminator) ...
    for (uint32_t i = 0; i < m_vehicles.GetN(); ++i) {
      Ptr<Socket> s = Socket::CreateSocket(m_vehicles.Get(i),
                                           TypeId::LookupByName("ns3::UdpSocketFactory"));
      s->Bind(InetSocketAddress(Ipv4Address::GetAny(), 8080));
      s->SetRecvCallback(MakeCallback(&FederatedRound::OnVehicleRx, this));
      m_vehSockets.push_back(s);
    }
    // ... and one per RSU (generator) to collect feedback.
    for (uint32_t i = 0; i < m_rsus.GetN(); ++i) {
      Ptr<Socket> s = Socket::CreateSocket(m_rsus.Get(i),
                                           TypeId::LookupByName("ns3::UdpSocketFactory"));
      s->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9090));
      s->SetRecvCallback(MakeCallback(&FederatedRound::OnRsuRx, this));
      m_rsuSockets.push_back(s);
    }
  }

  void Start() {
    std::ofstream f(m_out.c_str());
    f << "round,rsu,k_selected,payload_bytes,latency_s,bytes_down,bytes_up,"
         "total_bytes,replies_received,replies_expected\n";
    f.close();
    Simulator::Schedule(Seconds(1.0), &FederatedRound::RunRound, this, 0u);
  }

private:
  // The generator broadcasts its synthetic batch to a selected subset.
  void RunRound(uint32_t round) {
    if (round >= m_rounds) {
      return;
    }
    for (uint32_t g = 0; g < m_rsus.GetN(); ++g) {
      uint32_t perRsu = m_vehicles.GetN() / std::max<uint32_t>(m_rsus.GetN(), 1);
      uint32_t k = std::min(m_k, perRsu);

      // Select k distinct vehicles from this RSU's local pool.
      std::vector<uint32_t> pool;
      for (uint32_t i = 0; i < perRsu; ++i) {
        pool.push_back(g * perRsu + i);
      }
      for (uint32_t i = 0; i < k && pool.size() > 0; ++i) {
        uint32_t idx = m_rng->GetInteger(0, pool.size() - 1);
        std::swap(pool[idx], pool[pool.size() - 1]);
        uint32_t veh = pool.back();
        pool.pop_back();

        RoundKey key{round, g};
        if (m_stats[key].repliesExpected == 0) {
          m_stats[key].startTime = Simulator::Now().GetSeconds();
        }
        m_stats[key].repliesExpected += 1;
        m_stats[key].bytesDown += m_payload;
        // A vehicle may still owe a reply from an earlier round; queue rather
        // than overwrite, so no request is silently dropped.
        m_pendingQ[veh].push_back(key);

        // Hold a reference: closing the socket in the same event as Send()
        // can tear it down before the packet leaves the node.
        Ptr<Socket> s = Socket::CreateSocket(m_rsus.Get(g),
                                             TypeId::LookupByName("ns3::UdpSocketFactory"));
        s->Connect(InetSocketAddress(m_vehIf.GetAddress(veh), 8080));
        s->Send(Create<Packet>(m_payload));
        m_txSockets.push_back(s);
      }
    }
    Simulator::Schedule(Seconds(m_interval), &FederatedRound::RunRound, this, round + 1);
    // Flush the round's record once every reply has had time to arrive.
    Simulator::Schedule(Seconds(m_interval * 0.95), &FederatedRound::FlushRound, this, round);
  }

  // A discriminator received samples: it replies with critic feedback.
  void OnVehicleRx(Ptr<Socket> socket) {
    Ptr<Packet> p;
    Address from;
    while ((p = socket->RecvFrom(from))) {
      uint32_t node = socket->GetNode()->GetId();
      uint32_t veh = node >= m_rsus.GetN() ? node - m_rsus.GetN() : 0;
      auto it = m_pendingQ.find(veh);
      if (it == m_pendingQ.end() || it->second.empty()) {
        continue;
      }
      RoundKey key = it->second.front();
      it->second.pop_front();
      uint32_t g = key.rsu;
      Ptr<Socket> s = Socket::CreateSocket(socket->GetNode(),
                                           TypeId::LookupByName("ns3::UdpSocketFactory"));
      s->Connect(InetSocketAddress(m_rsuIf.GetAddress(g), 9090));
      // Carry the round in the reply so the generator can attribute it.
      ReplyTag tag;
      tag.round = key.round;
      Ptr<Packet> reply = Create<Packet>(m_payload);
      reply->AddPacketTag(tag);
      s->Send(reply);
      m_txSockets.push_back(s);
      m_stats[key].bytesUp += m_payload;
    }
  }

  // The generator received feedback: the round closes on the last arrival.
  void OnRsuRx(Ptr<Socket> socket) {
    Ptr<Packet> p;
    Address from;
    while ((p = socket->RecvFrom(from))) {
      uint32_t g = socket->GetNode()->GetId();
      double now = Simulator::Now().GetSeconds();
      ReplyTag tag;
      if (!p->PeekPacketTag(tag)) {
        continue;
      }
      RoundKey key{tag.round, g};
      auto it = m_stats.find(key);
      if (it == m_stats.end()) {
        continue;
      }
      it->second.repliesReceived += 1;
      // A synchronisation cycle closes when the slowest selected discriminator
      // replies, so the round's latency is the time of the *last* arrival.
      it->second.lastReplyTime = now;
    }
  }

  void FlushRound(uint32_t round) {
    std::ofstream f(m_out.c_str(), std::ios::app);
    for (uint32_t g = 0; g < m_rsus.GetN(); ++g) {
      RoundKey key{round, g};
      auto it = m_stats.find(key);
      if (it == m_stats.end()) {
        continue;
      }
      const RoundStat &s = it->second;
      double lat = s.repliesReceived ? (s.lastReplyTime - s.startTime) : -1.0;
      f << round << "," << g << "," << m_k << "," << m_payload << ","
        << lat << "," << s.bytesDown << "," << s.bytesUp << ","
        << (s.bytesDown + s.bytesUp) << "," << s.repliesReceived << ","
        << s.repliesExpected << "\n";
    }
  }

  struct RoundKey {
    uint32_t round;
    uint32_t rsu;
    bool operator<(const RoundKey &o) const {
      return round != o.round ? round < o.round : rsu < o.rsu;
    }
  };

  uint32_t m_k, m_payload, m_rounds;
  double m_interval;
  std::string m_out;
  uint32_t m_seed;
  NodeContainer m_rsus, m_vehicles;
  Ipv4InterfaceContainer m_rsuIf, m_vehIf;
  std::vector<Ptr<Socket>> m_vehSockets, m_rsuSockets, m_txSockets;
  std::map<RoundKey, RoundStat> m_stats;
  std::map<uint32_t, std::deque<RoundKey>> m_pendingQ;
  Ptr<UniformRandomVariable> m_rng;
};

}  // namespace

int main(int argc, char *argv[]) {
  uint32_t nRsu = 4, nVehicles = 20, kSelected = 3, rounds = 60;
  uint32_t payloadBytes = 1024, seed = 1;
  double interval = 1.0, simArea = 1500.0, speedMin = 15.0, speedMax = 60.0;
  double pathLossExp = 2.0;
  std::string out = "mdwgan_comm.csv";

  CommandLine cmd(__FILE__);
  cmd.AddValue("nRsu", "Number of RSUs (generators)", nRsu);
  cmd.AddValue("nVehicles", "Number of vehicles (discriminators)", nVehicles);
  cmd.AddValue("kSelected", "Discriminators selected per RSU per round", kSelected);
  cmd.AddValue("rounds", "Federated rounds to simulate", rounds);
  cmd.AddValue("payloadBytes", "Bytes per message", payloadBytes);
  cmd.AddValue("interval", "Seconds between rounds", interval);
  cmd.AddValue("speedMin", "Minimum vehicle speed (m/s)", speedMin);
  cmd.AddValue("speedMax", "Maximum vehicle speed (m/s)", speedMax);
  cmd.AddValue("simArea", "Side of the square deployment area (m)", simArea);
  cmd.AddValue("pathLossExp", "Log-distance path loss exponent", pathLossExp);
  cmd.AddValue("seed", "RNG run number", seed);
  cmd.AddValue("out", "Output CSV path", out);
  cmd.Parse(argc, argv);

  RngSeedManager::SetSeed(12345);
  RngSeedManager::SetRun(seed);

  NodeContainer rsus;
  rsus.Create(nRsu);
  NodeContainer vehicles;
  vehicles.Create(nVehicles);

  // IEEE 802.11p / DSRC at 10 MHz, as in Table II of the paper.
  // Build the channel from scratch rather than from Default(): Default()
  // already installs a Friis loss model, and AddPropagationLoss *chains* onto
  // it rather than replacing it, which double-counts the attenuation (~98 dB
  // of extra loss, putting every link below receiver sensitivity).
  YansWifiChannelHelper channel;
  channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  // Log-distance path loss (Eq. 2).  ReferenceLoss is the free-space loss at
  // 1 m for the 5.9 GHz DSRC band, 20log10(4*pi*d*f/c) = 47.86 dB -- ns-3's
  // 46.6777 default is the 2.4 GHz figure and under-models this band.
  channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                             "Exponent", DoubleValue(pathLossExp),
                             "ReferenceDistance", DoubleValue(1.0),
                             "ReferenceLoss", DoubleValue(47.86));
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());
  // Leave ChannelSettings at the default: WIFI_STANDARD_80211p selects the
  // correct 5.9 GHz 10 MHz control channel itself.  Forcing {0,10,BAND_5GHZ,0}
  // puts the PHY on a band where no frame is ever received.

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211p);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue("OfdmRate6MbpsBW10MHz"),
                               "ControlMode", StringValue("OfdmRate6MbpsBW10MHz"));
  WifiMacHelper mac;
  mac.SetType("ns3::AdhocWifiMac");

  NodeContainer all(rsus, vehicles);
  NetDeviceContainer devices = wifi.Install(phy, mac, all);

  // RSUs are stationary and evenly spaced; vehicles move within the area.
  MobilityHelper rsuMobility;
  Ptr<ListPositionAllocator> rsuPos = CreateObject<ListPositionAllocator>();
  for (uint32_t i = 0; i < nRsu; ++i) {
    rsuPos->Add(Vector(simArea * (i + 0.5) / nRsu, simArea / 2.0, 10.0));
  }
  rsuMobility.SetPositionAllocator(rsuPos);
  rsuMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  rsuMobility.Install(rsus);

  // RandomWaypoint needs a *constructed* position allocator object, not a
  // string: the same allocator seeds the initial placement and supplies each
  // subsequent waypoint.
  std::ostringstream box;
  box << "ns3::UniformRandomVariable[Min=0|Max=" << simArea << "]";
  Ptr<RandomBoxPositionAllocator> vehPos = CreateObject<RandomBoxPositionAllocator>();
  vehPos->SetAttribute("X", StringValue(box.str()));
  vehPos->SetAttribute("Y", StringValue(box.str()));
  vehPos->SetAttribute("Z", StringValue("ns3::ConstantRandomVariable[Constant=1.5]"));

  std::ostringstream speed;
  speed << "ns3::UniformRandomVariable[Min=" << speedMin << "|Max=" << speedMax << "]";

  MobilityHelper vehMobility;
  vehMobility.SetPositionAllocator(vehPos);
  vehMobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
      "Speed", StringValue(speed.str()),
      "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.2]"),
      "PositionAllocator", PointerValue(vehPos));
  vehMobility.Install(vehicles);

  InternetStackHelper internet;
  internet.Install(all);

  Ipv4AddressHelper addr;
  addr.SetBase("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer ifaces = addr.Assign(devices);

  Ipv4InterfaceContainer rsuIf, vehIf;
  for (uint32_t i = 0; i < nRsu; ++i) {
    rsuIf.Add(ifaces.Get(i));
  }
  for (uint32_t i = 0; i < nVehicles; ++i) {
    vehIf.Add(ifaces.Get(nRsu + i));
  }

  FederatedRound fed(kSelected, payloadBytes, rounds, interval, out, seed);
  fed.Setup(rsus, vehicles, rsuIf, vehIf);
  fed.Start();

  Simulator::Stop(Seconds(rounds * interval + 5.0));
  Simulator::Run();
  Simulator::Destroy();

  std::cout << "mdwgan-comm: wrote " << out << std::endl;
  return 0;
}
