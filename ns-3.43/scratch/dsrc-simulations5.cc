#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/netanim-module.h"
#include "ns3/olsr-helper.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("HybridGanDsrc");

const uint32_t NUM_HYBRID_NODES = 20;  // Each node is both generator and discriminator
const double SIMULATION_AREA_SIZE = 1000.0;
const double LOSS_CONVERGENCE_THRESHOLD = 0.005;
const double MAX_SIM_TIME = 150.0;

struct HybridLossTracker {
    std::map<uint32_t, double> generatorLossValues;
    std::map<uint32_t, double> discriminatorLossValues;
    std::map<uint32_t, double> prevGeneratorLossValues;
    std::map<uint32_t, double> prevDiscriminatorLossValues;
    
    HybridLossTracker() {
        for (uint32_t i = 1; i <= NUM_HYBRID_NODES; i++) {
            generatorLossValues[i] = 9999.0;
            discriminatorLossValues[i] = 9999.0;
            prevGeneratorLossValues[i] = 9999.0;
            prevDiscriminatorLossValues[i] = 9999.0;
        }
    }
    
    bool updateGeneratorLoss(uint32_t nodeId, double newLoss) {
        prevGeneratorLossValues[nodeId] = generatorLossValues[nodeId];
        if (newLoss < generatorLossValues[nodeId]) {
            generatorLossValues[nodeId] = newLoss;
            return true;
        }
        return false;
    }
    
    bool updateDiscriminatorLoss(uint32_t nodeId, double newLoss) {
        prevDiscriminatorLossValues[nodeId] = discriminatorLossValues[nodeId];
        if (newLoss < discriminatorLossValues[nodeId]) {
            discriminatorLossValues[nodeId] = newLoss;
            return true;
        }
        return false;
    }
    
    double getGeneratorLoss(uint32_t nodeId) {
        return generatorLossValues[nodeId];
    }
    
    double getDiscriminatorLoss(uint32_t nodeId) {
        return discriminatorLossValues[nodeId];
    }
    
    bool hasConverged() {
        double maxGenDelta = 0.0;
        double maxDiscDelta = 0.0;
        
        for (uint32_t i = 1; i <= NUM_HYBRID_NODES; i++) {
            double genDelta = std::abs(generatorLossValues[i] - prevGeneratorLossValues[i]);
            double discDelta = std::abs(discriminatorLossValues[i] - prevDiscriminatorLossValues[i]);
            if (genDelta > maxGenDelta) maxGenDelta = genDelta;
            if (discDelta > maxDiscDelta) maxDiscDelta = discDelta;
        }
        
        bool converged = (maxGenDelta < LOSS_CONVERGENCE_THRESHOLD) && 
                        (maxDiscDelta < LOSS_CONVERGENCE_THRESHOLD) && 
                        (maxDiscDelta != 0);
        if (converged) {
            NS_LOG_UNCOND("Hybrid GAN training has converged. Max generator loss change: " << maxGenDelta
                         << ", Max discriminator loss change: " << maxDiscDelta);
        }
        return converged;
    }
} hybridLossTracker;

struct NetworkMetrics {
    std::map<uint32_t, double> latencies;
    std::map<uint32_t, uint32_t> packetLosses;
    
    void recordLatency(uint32_t nodeId, double latency) {
        latencies[nodeId] = latency;
    }
    
    void recordPacketLoss(uint32_t nodeId) {
        packetLosses[nodeId]++;
    }
    
    double getAvgLatency() {
        double sum = 0.0;
        uint32_t count = 0;
        for (const auto& pair : latencies) {
            sum += pair.second;
            count++;
        }
        return count > 0 ? sum / count : 0.0;
    }
    
    uint32_t getTotalPacketLoss() {
        uint32_t sum = 0;
        for (const auto& pair : packetLosses) {
            sum += pair.second;
        }
        return sum;
    }
} networkMetrics;

bool CheckHybridGanEnvironment() {
    std::string scriptPath1 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_hybrid_node.py";
    std::string scriptPath2 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/ddpg_agent.py";
    std::string scriptPath3 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/plot_resource_usage.py";
    
    std::ifstream file1(scriptPath1.c_str());
    std::ifstream file2(scriptPath2.c_str());
    std::ifstream file3(scriptPath3.c_str());
    
    bool scriptsExist = file1.good() && file2.good() && file3.good();
    file1.close();
    file2.close();
    file3.close();
    
    if (!scriptsExist) {
        NS_LOG_ERROR("Hybrid GAN Python scripts not found!");
        return false;
    }
    
    std::string baseDir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/";
    std::vector<std::string> dirs;
    for (uint32_t i = 1; i <= NUM_HYBRID_NODES; i++) {
        dirs.push_back(baseDir + "results/hybrid_node_" + std::to_string(i));
    }
    dirs.push_back(baseDir + "results");
    
    for (const std::string& dir : dirs) {
        std::string mkdirCmd = "mkdir -p " + dir;
        int status = system(mkdirCmd.c_str());
        if (status != 0) {
            NS_LOG_ERROR("Failed to create directory: " << dir);
            return false;
        }
    }
    
    std::string lossFilePath = baseDir + "results/hybrid_loss_tracking.csv";
    std::ofstream lossFile(lossFilePath.c_str());
    if (lossFile.is_open()) {
        lossFile << "Timestamp,NodeID,GeneratorLoss,DiscriminatorLoss,GenImproved,DiscImproved\n";
        lossFile.close();
    } else {
        NS_LOG_ERROR("Failed to create loss tracking file: " << lossFilePath);
        return false;
    }
    
    return true;
}

Ptr<Packet> CreatePacketWithData(const std::string& data) {
    uint8_t *buffer = new uint8_t[data.length() + 1];
    memcpy(buffer, data.c_str(), data.length() + 1);
    
    Ptr<Packet> packet = Create<Packet>(buffer, data.length() + 1);
    delete[] buffer;
    
    return packet;
}

std::string ExtractDataFromPacket(Ptr<Packet> packet) {
    uint8_t *buffer = new uint8_t[packet->GetSize()];
    packet->CopyData(buffer, packet->GetSize());
    
    std::string data((char*)buffer, packet->GetSize());
    delete[] buffer;
    
    return data;
}

void LogHybridLossUpdate(uint32_t nodeId, double genLoss, double discLoss, bool genImproved, bool discImproved) {
    std::string baseDir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/";
    std::string lossFilePath = baseDir + "results/hybrid_loss_tracking.csv";
    
    std::ofstream lossFile(lossFilePath.c_str(), std::ios::app);
    if (lossFile.is_open()) {
        time_t now = time(NULL);
        lossFile << now << "," << nodeId << "," << genLoss << "," << discLoss 
                << "," << (genImproved ? "Yes" : "No") << "," << (discImproved ? "Yes" : "No") << "\n";
        lossFile.close();
    } else {
        NS_LOG_ERROR("Failed to open hybrid loss tracking file");
    }
}

void RunDDPGAgent() {
    std::string cmd = "python3 /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/ddpg_agent.py > /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/ddpg_log.txt 2>&1";
    NS_LOG_UNCOND("Executing DDPG agent command: " << cmd);
    int status = system(cmd.c_str());
    if (status != 0) {
        NS_LOG_ERROR("Failed to run DDPG agent: " << status);
    } else {
        NS_LOG_INFO("DDPG agent executed successfully");
    }
}

void PlotResourceUsage() {
    std::string cmd = "python3 /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/plot_resource_usage.py > /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/plot_log.txt 2>&1";
    NS_LOG_UNCOND("Generating resource usage charts: " << cmd);
    int status = system(cmd.c_str());
    if (status != 0) {
        NS_LOG_ERROR("Failed to generate resource usage charts: " << status);
    } else {
        NS_LOG_INFO("Resource usage charts generated successfully");
    }
}

void CustomHybridReceiveCallback(Ptr<Socket> socket) {
    if (!socket) {
        NS_LOG_ERROR("Invalid socket pointer");
        return;
    }
    
    Ptr<Packet> packet;
    Address from;
    Time receiveTime = Simulator::Now();
    
    while ((packet = socket->RecvFrom(from))) {
        uint32_t nodeId = socket->GetNode()->GetId();
        InetSocketAddress inetFrom = InetSocketAddress::ConvertFrom(from);
        Ipv4Address senderAddr = inetFrom.GetIpv4();
        
        double latency = (Simulator::Now() - receiveTime).GetSeconds();
        networkMetrics.recordLatency(nodeId, latency);
        
        std::ostringstream addrStream;
        senderAddr.Print(addrStream);
        std::string senderAddrStr = addrStream.str();
        
        NS_LOG_INFO("Hybrid node " << nodeId << " received packet from " << senderAddr << " of size " << packet->GetSize());
        
        std::string receivedData = ExtractDataFromPacket(packet);
        NS_LOG_UNCOND("Extracted data: " << receivedData);
        
        if (nodeId >= NUM_HYBRID_NODES) {
            NS_LOG_UNCOND("Node " << nodeId << " not configured for hybrid GAN training");
            continue;
        }
        
        // Parse received data (expected format: "gen_loss:X.X,disc_loss:Y.Y")
        double genLoss = 9999.0, discLoss = 9999.0;
        size_t genPos = receivedData.find("gen_loss:");
        size_t discPos = receivedData.find("disc_loss:");
        
        if (genPos != std::string::npos) {
            size_t commaPos = receivedData.find(",", genPos);
            if (commaPos != std::string::npos) {
                genLoss = std::stod(receivedData.substr(genPos + 9, commaPos - genPos - 9));
            }
        }
        
        if (discPos != std::string::npos) {
            discLoss = std::stod(receivedData.substr(discPos + 10));
        }
        
        bool genImproved = hybridLossTracker.updateGeneratorLoss(nodeId + 1, genLoss);
        bool discImproved = hybridLossTracker.updateDiscriminatorLoss(nodeId + 1, discLoss);
        
        LogHybridLossUpdate(nodeId + 1, genLoss, discLoss, genImproved, discImproved);
        
        if (genImproved || discImproved) {
            NS_LOG_UNCOND("Hybrid node " << (nodeId + 1) << " - Generator loss: " << genLoss 
                        << " (improved: " << genImproved << "), Discriminator loss: " << discLoss 
                        << " (improved: " << discImproved << ")");
        }
        
        // Check for convergence
        if (hybridLossTracker.hasConverged()) {
            NS_LOG_UNCOND("Stopping simulation due to hybrid GAN convergence at time " << Simulator::Now().GetSeconds() << " seconds");
            Simulator::Stop();
            PlotResourceUsage();
            return;
        }
        
        // Run DDPG agent periodically
        static uint32_t ddpgCounter = 0;
        ddpgCounter++;
        if (ddpgCounter % 5 == 0) {  // Run every 5 received packets
            RunDDPGAgent();
        }
        
        // Execute hybrid node script
        std::ostringstream cmd;
        uint32_t hybridNodeId = nodeId + 1;
        double currentGenLoss = hybridLossTracker.getGeneratorLoss(hybridNodeId);
        double currentDiscLoss = hybridLossTracker.getDiscriminatorLoss(hybridNodeId);
        
        cmd << "python3 /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_hybrid_node.py " 
            << hybridNodeId << " " << senderAddr.Get() << " " << currentGenLoss << " " << currentDiscLoss
            << " > /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/hybrid_node_" 
            << hybridNodeId << "/log_" << time(NULL) << ".txt 2>&1";
            
        NS_LOG_UNCOND("Running hybrid node " << hybridNodeId << " with command: " << cmd.str());
        
        std::string bgCmd = cmd.str() + " &";
        std::string scriptPath = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_hybrid_node.py";
        
        std::ifstream scriptFile(scriptPath.c_str());
        if (!scriptFile.good()) {
            NS_LOG_ERROR("Python script not found: " << scriptPath);
            continue;
        }
        scriptFile.close();
        
        try {
            int status = system(bgCmd.c_str());
            if (status != 0) {
                NS_LOG_ERROR("Failed to start hybrid GAN script: " << status);
            } else {
                NS_LOG_UNCOND("Hybrid GAN script started in background");
            }
        } catch (const std::exception& e) {
            NS_LOG_ERROR("Exception during command execution: " << e.what());
        }
    }
}

double GenerateInitialHybridLoss() {
    srand(time(NULL));
    return 2.0 + (rand() % 1000) / 100.0;
}

void SendPacketWithHybridData(Ptr<Socket> socket, Ptr<Node> targetNode, double genLoss, double discLoss) {
    Ipv4Address targetAddress = targetNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    
    std::ostringstream dataStream;
    dataStream << "gen_loss:" << genLoss << ",disc_loss:" << discLoss;
    std::string data = dataStream.str();
    
    Ptr<Packet> packet = CreatePacketWithData(data);
    socket->SendTo(packet, 0, InetSocketAddress(targetAddress, 9));
    
    NS_LOG_UNCOND("Sent hybrid data " << data << " to node " 
                 << targetNode->GetId() << " at " << targetAddress);
}

int main(int argc, char *argv[])
{
    uint32_t nVehicles = 100;
    double simTime = MAX_SIM_TIME;
    bool verbose = true;
    
    CommandLine cmd(__FILE__);
    cmd.AddValue("nVehicles", "Number of vehicles", nVehicles);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("verbose", "Enable verbose logging", verbose);
    cmd.Parse(argc, argv);
    
    if (!CheckHybridGanEnvironment()) {
        NS_LOG_ERROR("Hybrid GAN environment setup issues detected.");
        return 1;
    }
    
    if (verbose) {
        LogComponentEnable("HybridGanDsrc", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("Socket", LOG_LEVEL_INFO);
    }
    
    if (nVehicles < NUM_HYBRID_NODES + 5) {
        nVehicles = NUM_HYBRID_NODES + 5;
        NS_LOG_WARN("Increasing number of vehicles to " << nVehicles);
    }
    
    NS_LOG_INFO("Creating " << nVehicles << " vehicles with " << NUM_HYBRID_NODES << " hybrid GAN nodes");
    NodeContainer vehicles;
    vehicles.Create(nVehicles);
    
    // WiFi setup (same as original)
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    
    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent", DoubleValue(3.0));
    
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(23));
    wifiPhy.Set("TxPowerEnd", DoubleValue(23));
    wifiPhy.Set("TxGain", DoubleValue(5));
    wifiPhy.Set("RxGain", DoubleValue(5));
    
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue("OfdmRate6MbpsBW10MHz"),
                               "ControlMode", StringValue("OfdmRate6MbpsBW10MHz"));
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, vehicles);
    
    // Position allocation for hybrid nodes in clusters
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    
    // Arrange hybrid nodes in 4 clusters of 5 nodes each
    uint32_t clustersPerSide = 2;
    uint32_t nodesPerCluster = 5;
    double clusterSpacing = SIMULATION_AREA_SIZE / (clustersPerSide + 1);
    
    for (uint32_t cluster = 0; cluster < 4; cluster++) {
        uint32_t clusterRow = cluster / clustersPerSide;
        uint32_t clusterCol = cluster % clustersPerSide;
        
        double clusterCenterX = (clusterCol + 1) * clusterSpacing;
        double clusterCenterY = (clusterRow + 1) * clusterSpacing;
        double radius = SIMULATION_AREA_SIZE * 0.05;
        
        for (uint32_t nodeInCluster = 0; nodeInCluster < nodesPerCluster; nodeInCluster++) {
            double angle = 2.0 * M_PI * nodeInCluster / nodesPerCluster;
            double x = clusterCenterX + radius * cos(angle);
            double y = clusterCenterY + radius * sin(angle);
            positionAlloc->Add(Vector(x, y, 0.0));
        }
    }
    
    // Random positions for remaining vehicles
    for (uint32_t i = NUM_HYBRID_NODES; i < nVehicles; i++) {
        double x = SIMULATION_AREA_SIZE * ((double) rand() / RAND_MAX);
        double y = SIMULATION_AREA_SIZE * ((double) rand() / RAND_MAX);
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    
    // Mobility setup
    MobilityHelper mobility;
    mobility.SetPositionAllocator(positionAlloc);
    
    // Hybrid nodes have limited mobility within their clusters
    for (uint32_t i = 0; i < NUM_HYBRID_NODES; i++) {
        uint32_t cluster = i / 5;
        uint32_t clusterRow = cluster / 2;
        uint32_t clusterCol = cluster % 2;
        
        double clusterCenterX = (clusterCol + 1) * (SIMULATION_AREA_SIZE / 3);
        double clusterCenterY = (clusterRow + 1) * (SIMULATION_AREA_SIZE / 3);
        double clusterSize = SIMULATION_AREA_SIZE * 0.15;
        
        mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                 "Bounds", RectangleValue(Rectangle(
                                     clusterCenterX - clusterSize/2, clusterCenterX + clusterSize/2,
                                     clusterCenterY - clusterSize/2, clusterCenterY + clusterSize/2)),
                                 "Speed", StringValue("ns3::UniformRandomVariable[Min=2|Max=8]"));
        mobility.Install(vehicles.Get(i));
    }
    
    // Other vehicles have full mobility
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(0, SIMULATION_AREA_SIZE, 0, SIMULATION_AREA_SIZE)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=10|Max=100]"));
    for (uint32_t i = NUM_HYBRID_NODES; i < nVehicles; i++) {
        mobility.Install(vehicles.Get(i));
    }
    
    // Internet and routing setup
    InternetStackHelper internet;
    OlsrHelper olsr;
    internet.SetRoutingHelper(olsr);
    internet.Install(vehicles);
    
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4h.Assign(devices);
    
    for (uint32_t i = 0; i < nVehicles; i++) {
        Ptr<Ipv4> ipv4Interface = vehicles.Get(i)->GetObject<Ipv4>();
        ipv4Interface->SetAttribute("IpForward", BooleanValue(true));
    }
    
    // Socket setup for hybrid nodes
    std::vector<Ptr<Socket>> hybridSockets;
    uint16_t basePort = 9;
    
    for (uint32_t i = 0; i < NUM_HYBRID_NODES; i++) {
        TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
        Ptr<Socket> recvSocket = Socket::CreateSocket(vehicles.Get(i), tid);
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), basePort);
        if (recvSocket->Bind(local) == -1) {
            NS_LOG_ERROR("Failed to bind socket on hybrid node " << i);
            return -1;
        }
        recvSocket->SetRecvCallback(MakeCallback(&CustomHybridReceiveCallback));
        NS_LOG_INFO("Socket bound on hybrid node " << i << " to port " << basePort);
        hybridSockets.push_back(recvSocket);
    }
    
    // Initialize loss values for all hybrid nodes
    for (uint32_t i = 1; i <= NUM_HYBRID_NODES; i++) {
        double initialGenLoss = GenerateInitialHybridLoss();
        double initialDiscLoss = GenerateInitialHybridLoss();
        hybridLossTracker.updateGeneratorLoss(i, initialGenLoss);
        hybridLossTracker.updateDiscriminatorLoss(i, initialDiscLoss);
    }
    
    // Communication setup between hybrid nodes
    ApplicationContainer hybridApps;
    for (uint32_t i = 0; i < NUM_HYBRID_NODES; i++) {
        for (uint32_t j = 0; j < NUM_HYBRID_NODES; j++) {
            if (i == j) continue;
            
            UdpEchoClientHelper client(interfaces.GetAddress(j), basePort);
            client.SetAttribute("MaxPackets", UintegerValue(50));
            client.SetAttribute("Interval", TimeValue(Seconds(15.0)));
            client.SetAttribute("PacketSize", UintegerValue(1024));
            
            ApplicationContainer app = client.Install(vehicles.Get(i));
            app.Start(Seconds(5.0 + i * 0.5));
            app.Stop(Seconds(simTime - 5.0));
            hybridApps.Add(app);
            
            // Schedule initial data transmission
            Simulator::Schedule(Seconds(5.0 + i * 0.5), 
                              &SendPacketWithHybridData, 
                              hybridSockets[i],
                              vehicles.Get(j),
                              hybridLossTracker.getGeneratorLoss(i + 1),
                              hybridLossTracker.getDiscriminatorLoss(i + 1));
        }
    }
    
    // Schedule periodic loss updates
    for (uint32_t i = 0; i < NUM_HYBRID_NODES; i++) {
        for (double t = 10.0; t < simTime - 10.0; t += 15.0) {
            double delay = t + (rand() % 5);
            
            Simulator::Schedule(Seconds(delay), 
                               [i, &vehicles, &hybridSockets]() {
                                   uint32_t nodeId = i + 1;
                                   double currentGenLoss = hybridLossTracker.getGeneratorLoss(nodeId);
                                   double currentDiscLoss = hybridLossTracker.getDiscriminatorLoss(nodeId);
                                   
                                   // Simulate loss improvement
                                   double genImprovementFactor = 0.85 + (rand() % 15) / 100.0;
                                   double discImprovementFactor = 0.85 + (rand() % 15) / 100.0;
                                   double newGenLoss = currentGenLoss * genImprovementFactor;
                                   double newDiscLoss = currentDiscLoss * discImprovementFactor;
                                   
                                   bool genImproved = hybridLossTracker.updateGeneratorLoss(nodeId, newGenLoss);
                                   bool discImproved = hybridLossTracker.updateDiscriminatorLoss(nodeId, newDiscLoss);
                                   
                                   if (genImproved || discImproved) {
                                       NS_LOG_UNCOND("Scheduling improved losses from hybrid node " << nodeId 
                                                    << " - Gen: " << newGenLoss << " (" << genImproved 
                                                    << "), Disc: " << newDiscLoss << " (" << discImproved << ")");
                                       LogHybridLossUpdate(nodeId, newGenLoss, newDiscLoss, genImproved, discImproved);
                                       
                                       // Send updated losses to other nodes
                                       for (uint32_t j = 0; j < NUM_HYBRID_NODES; j++) {
                                           if (j != i) {
                                               SendPacketWithHybridData(hybridSockets[i], vehicles.Get(j), newGenLoss, newDiscLoss);
                                           }
                                       }
                                   }
                               });
        }
    }
    
    // Background traffic from regular vehicles
    ApplicationContainer backgroundApps;
    for (uint32_t i = NUM_HYBRID_NODES; i < std::min(nVehicles, NUM_HYBRID_NODES + 15); i++) {
        uint32_t dest = (i + 1) % nVehicles;
        if (dest < NUM_HYBRID_NODES) dest = NUM_HYBRID_NODES;
        
        UdpEchoClientHelper client(interfaces.GetAddress(dest), basePort + 1);
        client.SetAttribute("MaxPackets", UintegerValue(10));
        client.SetAttribute("Interval", TimeValue(Seconds(120.0)));
        client.SetAttribute("PacketSize", UintegerValue(512));
        
        ApplicationContainer app = client.Install(vehicles.Get(i));
        app.Start(Seconds(10.0 + (i - NUM_HYBRID_NODES) * 0.2));
        app.Stop(Seconds(simTime - 10.0));
        backgroundApps.Add(app);
    }
    
    // PCAP tracing for hybrid nodes
    for (uint32_t i = 0; i < NUM_HYBRID_NODES; i++) {
        std::ostringstream pcapName;
        pcapName << "hybrid-gan-node-" << (i + 1);
        wifiPhy.EnablePcap(pcapName.str(), devices.Get(i));
    }
    
    // Animation setup
    AnimationInterface anim("hybrid-gan-animation.xml");
    anim.SetMaxPktsPerTraceFile(100000);
    
    // Color hybrid nodes in different clusters
    uint8_t clusterColors[][3] = {
        {255, 0, 0},   // Red cluster
        {0, 255, 0},   // Green cluster  
        {0, 0, 255},   // Blue cluster
        {255, 255, 0}  // Yellow cluster
    };
    
    for (uint32_t i = 0; i < NUM_HYBRID_NODES; i++) {
        uint32_t cluster = i / 5;
        anim.UpdateNodeColor(vehicles.Get(i), 
                            clusterColors[cluster][0], 
                            clusterColors[cluster][1], 
                            clusterColors[cluster][2]);
        anim.UpdateNodeSize(vehicles.Get(i), 12, 12);
        std::ostringstream desc;
        desc << "Hybrid Node " << (i + 1) << " (Cluster " << (cluster + 1) << ")";
        anim.UpdateNodeDescription(vehicles.Get(i), desc.str());
    }
    
    // Color regular vehicles in gray
    for (uint32_t i = NUM_HYBRID_NODES; i < nVehicles; i++) {
        anim.UpdateNodeColor(vehicles.Get(i), 180, 180, 180);
    }
    
    NS_LOG_INFO("Starting hybrid GAN simulation for up to " << simTime << " seconds");
    Simulator::Stop(Seconds(simTime + 1));
    Simulator::Run();
    Simulator::Destroy();
    
    NS_LOG_INFO("Hybrid GAN simulation completed");
    return 0;
}