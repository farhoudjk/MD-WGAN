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

NS_LOG_COMPONENT_DEFINE("DsrcHighwaySimple");

const uint32_t NUM_GENERATORS = 4;
const uint32_t NUM_DISCRIMINATORS = 20;
const uint32_t TOTAL_GAN_NODES = NUM_GENERATORS + NUM_DISCRIMINATORS;
const double SIMULATION_AREA_SIZE = 1000.0;
const double LOSS_CONVERGENCE_THRESHOLD = 0.005;  // Convergence criterion
const double MAX_SIM_TIME = 150.0;

struct LossTracker {
    std::map<uint32_t, double> generatorLossValues;
    std::map<uint32_t, double> discriminatorLossValues;
    std::map<uint32_t, double> prevGeneratorLossValues;
    std::map<uint32_t, double> prevDiscriminatorLossValues;
    
    LossTracker() {
        for (uint32_t i = 1; i <= NUM_GENERATORS; i++) {
            generatorLossValues[i] = 9999.0;
            prevGeneratorLossValues[i] = 9999.0;
        }
        for (uint32_t i = 1; i <= NUM_DISCRIMINATORS; i++) {
            discriminatorLossValues[i] = 9999.0;
            prevDiscriminatorLossValues[i] = 9999.0;
        }
    }
    
    bool updateGeneratorLoss(uint32_t genId, double newLoss) {
        prevGeneratorLossValues[genId] = generatorLossValues[genId];
        if (newLoss < generatorLossValues[genId]) {
            generatorLossValues[genId] = newLoss;
            return true;
        }
        return false;
    }
    
    bool updateDiscriminatorLoss(uint32_t discId, double newLoss) {
        prevDiscriminatorLossValues[discId] = discriminatorLossValues[discId];
        if (newLoss < discriminatorLossValues[discId]) {
            discriminatorLossValues[discId] = newLoss;
            return true;
        }
        return false;
    }
    
    double getGeneratorLoss(uint32_t genId) {
        return generatorLossValues[genId];
    }
    
    double getDiscriminatorLoss(uint32_t discId) {
        return discriminatorLossValues[discId];
    }
    
    bool hasConverged() {
        double maxGenDelta = 0.0;
        double maxDiscDelta = 0.0;
        
        for (uint32_t i = 1; i <= NUM_GENERATORS; i++) {
            double delta = std::abs(generatorLossValues[i] - prevGeneratorLossValues[i]);
            if (delta > maxGenDelta) maxGenDelta = delta;
        }
        
        for (uint32_t i = 1; i <= NUM_DISCRIMINATORS; i++) {
            double delta = std::abs(discriminatorLossValues[i] - prevDiscriminatorLossValues[i]);
            if (delta > maxDiscDelta) maxDiscDelta = delta;
        }
        
        bool converged = (maxGenDelta < LOSS_CONVERGENCE_THRESHOLD) && (maxDiscDelta < LOSS_CONVERGENCE_THRESHOLD) && (maxDiscDelta !=0);
        if (converged) {
            NS_LOG_UNCOND("GAN training has converged. Max generator loss change: " << maxGenDelta
                         << ", Max discriminator loss change: " << maxDiscDelta);
        }
        return converged;
    }
} lossTracker;

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

bool CheckGanEnvironment() {
    std::string scriptPath1 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_generator2.py";
    std::string scriptPath2 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_discriminator2.py";
    std::string scriptPath3 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/rl_agent.py";
    std::string scriptPath4 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/plot_resource_usage.py";
    
    std::ifstream file1(scriptPath1.c_str());
    std::ifstream file2(scriptPath2.c_str());
    std::ifstream file3(scriptPath3.c_str());
    std::ifstream file4(scriptPath4.c_str());
    
    bool scriptsExist = file1.good() && file2.good() && file3.good() && file4.good();
    file1.close();
    file2.close();
    file3.close();
    file4.close();
    
    if (!scriptsExist) {
        NS_LOG_ERROR("GAN Python scripts not found!");
        return false;
    }
    
    std::string baseDir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/";
    std::vector<std::string> dirs;
    for (uint32_t i = 1; i <= NUM_GENERATORS; i++) {
        dirs.push_back(baseDir + "results/generator_" + std::to_string(i));
    }
    for (uint32_t i = 1; i <= NUM_DISCRIMINATORS; i++) {
        dirs.push_back(baseDir + "results/discriminator_" + std::to_string(i));
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
    
    std::string lossFilePath = baseDir + "results/loss_tracking.csv";
    std::ofstream lossFile(lossFilePath.c_str());
    if (lossFile.is_open()) {
        lossFile << "Timestamp,NodeType,NodeID,LossValue,Improved\n";
        lossFile.close();
    } else {
        NS_LOG_ERROR("Failed to create loss tracking file: " << lossFilePath);
        return false;
    }
    
    return true;
}

uint32_t MapToDiscriminatorId(uint32_t nodeId) {
    if (nodeId >= NUM_GENERATORS && nodeId < TOTAL_GAN_NODES) {
        return nodeId - NUM_GENERATORS + 1;
    }
    return 0;
}

uint32_t GetGeneratorForDiscriminator(uint32_t nodeId) {
    if (nodeId >= NUM_GENERATORS && nodeId < NUM_GENERATORS + 5) return 0;
    else if (nodeId >= NUM_GENERATORS + 5 && nodeId < NUM_GENERATORS + 10) return 1;
    else if (nodeId >= NUM_GENERATORS + 10 && nodeId < NUM_GENERATORS + 15) return 2;
    else if (nodeId >= NUM_GENERATORS + 15 && nodeId < TOTAL_GAN_NODES) return 3;
    return 0;
}

Ptr<Packet> CreatePacketWithLoss(double lossValue) {
    std::ostringstream oss;
    oss << lossValue;
    std::string lossStr = oss.str();
    
    uint8_t *buffer = new uint8_t[lossStr.length() + 1];
    memcpy(buffer, lossStr.c_str(), lossStr.length() + 1);
    
    Ptr<Packet> packet = Create<Packet>(buffer, lossStr.length() + 1);
    delete[] buffer;
    
    return packet;
}

double ExtractLossFromPacket(Ptr<Packet> packet) {
    uint8_t *buffer = new uint8_t[packet->GetSize()];
    packet->CopyData(buffer, packet->GetSize());
    
    std::string lossStr((char*)buffer, packet->GetSize());
    delete[] buffer;
    
    try {
        return std::stod(lossStr);
    } catch (const std::exception& e) {
        NS_LOG_ERROR("Failed to parse loss value: " << lossStr);
        return 9999.0;
    }
}

void LogLossUpdate(const std::string& nodeType, uint32_t nodeId, double lossValue, bool improved) {
    std::string baseDir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/";
    std::string lossFilePath = baseDir + "results/loss_tracking.csv";
    
    std::ofstream lossFile(lossFilePath.c_str(), std::ios::app);
    if (lossFile.is_open()) {
        time_t now = time(NULL);
        lossFile << now << "," << nodeType << "," << nodeId << "," << lossValue << "," << (improved ? "Yes" : "No") << "\n";
        lossFile.close();
    } else {
        NS_LOG_ERROR("Failed to open loss tracking file");
    }
}

void LogNetworkMetrics(uint32_t nodeId, double latency, bool packetLost) {
    std::string baseDir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/";
    std::string metricsFilePath = baseDir + "results/network_metrics.csv";
    
    std::ofstream metricsFile(metricsFilePath.c_str(), std::ios::app);
    if (metricsFile.is_open()) {
        time_t now = time(NULL);
        metricsFile << now << "," << nodeId << "," << latency << "," << (packetLost ? 1 : 0) << "\n";
        metricsFile.close();
    }
}

void RunRLAgent() {
    std::string cmd = "python3 /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/rl_agent.py > /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/rl_log.txt 2>&1";
    NS_LOG_UNCOND("Executing RL agent command: " << cmd);
    int status = system(cmd.c_str());
    if (status != 0) {
        NS_LOG_ERROR("Failed to run RL agent: " << status);
    } else {
        NS_LOG_INFO("RL agent executed successfully");
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

void CustomServerReceiveCallback(Ptr<Socket> socket) {
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
        
        NS_LOG_INFO("Node " << nodeId << " received packet from " << senderAddr << " of size " << packet->GetSize());
        
        double receivedLoss = ExtractLossFromPacket(packet);
        NS_LOG_UNCOND("Extracted loss value: " << receivedLoss);
        
        if (nodeId >= TOTAL_GAN_NODES) {
            NS_LOG_UNCOND("Node " << nodeId << " not configured for GAN training");
            continue;
        }
        
        bool improvedLoss = false;
        double currentBestLoss = 9999.0;
        
        if (nodeId < NUM_GENERATORS) {
            uint32_t genId = nodeId + 1;
            improvedLoss = lossTracker.updateGeneratorLoss(genId, receivedLoss);
            currentBestLoss = lossTracker.getGeneratorLoss(genId);
            
            LogLossUpdate("Generator", genId, receivedLoss, improvedLoss);
            
            if (improvedLoss) {
                NS_LOG_UNCOND("Generator " << genId << " loss improved to " << currentBestLoss);
            } else {
                NS_LOG_UNCOND("Generator " << genId << " received loss " << receivedLoss 
                            << " (no improvement: " << currentBestLoss << ")");
            }
        } else if (nodeId >= NUM_GENERATORS && nodeId < TOTAL_GAN_NODES) {
            uint32_t discId = MapToDiscriminatorId(nodeId);
            improvedLoss = lossTracker.updateDiscriminatorLoss(discId, receivedLoss);
            currentBestLoss = lossTracker.getDiscriminatorLoss(discId);
            
            LogLossUpdate("Discriminator", discId, receivedLoss, improvedLoss);
            
            if (improvedLoss) {
                NS_LOG_UNCOND("Discriminator " << discId << " loss improved to " << currentBestLoss);
            } else {
                NS_LOG_UNCOND("Discriminator " << discId << " received loss " << receivedLoss 
                            << " (no improvement: " << currentBestLoss << ")");
            }
        }
        
        // Check for convergence
        if (lossTracker.hasConverged()) {
            NS_LOG_UNCOND("Stopping simulation due to GAN convergence at time " << Simulator::Now().GetSeconds() << " seconds");
            Simulator::Stop();
            PlotResourceUsage();
            return;
        }
        
        // Run RL agent before executing generator scripts
        if (nodeId < NUM_GENERATORS) {
            RunRLAgent();
        }
        
        std::ostringstream cmd;
        if (nodeId < NUM_GENERATORS) {
            uint32_t genId = nodeId + 1;
            cmd << "python3 /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_generator2.py " 
                << genId << " " << senderAddr.Get() << " " << currentBestLoss
                << " > /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator_" 
                << genId << "/log_" << time(NULL) << ".txt 2>&1";
                
            NS_LOG_UNCOND("Running generator " << genId << " with command: " << cmd.str());
        } else if (nodeId >= NUM_GENERATORS && nodeId < TOTAL_GAN_NODES) {
            uint32_t discId = MapToDiscriminatorId(nodeId);
            uint32_t genId = GetGeneratorForDiscriminator(nodeId) + 1;
            
            cmd << "python3 /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_discriminator2.py " 
                << discId << " " << senderAddr.Get() << " " << genId << " " << currentBestLoss
                << " > /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_" 
                << discId << "/log_" << time(NULL) << ".txt 2>&1";
                
            NS_LOG_UNCOND("Running discriminator " << discId << " for generator " 
                        << genId << " with command: " << cmd.str());
        } else {
            NS_LOG_UNCOND("Node " << nodeId << " not configured for GAN training");
            continue;
        }
        
        std::string bgCmd = cmd.str() + " &";
        std::string scriptPath = (nodeId < NUM_GENERATORS) ? 
            "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_generator2.py" : 
            "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_discriminator2.py";
        
        std::ifstream scriptFile(scriptPath.c_str());
        if (!scriptFile.good()) {
            NS_LOG_ERROR("Python script not found: " << scriptPath);
            continue;
        }
        scriptFile.close();
        
        try {
            int status = system(bgCmd.c_str());
            if (status != 0) {
                NS_LOG_ERROR("Failed to start GAN script: " << status);
            } else {
                NS_LOG_UNCOND("GAN script started in background");
            }
        } catch (const std::exception& e) {
            NS_LOG_ERROR("Exception during command execution: " << e.what());
        }
        
        LogNetworkMetrics(nodeId, latency, false);
    }
}

double GenerateInitialLoss(uint32_t nodeId) {
    srand(time(NULL) + nodeId);
    return 2.0 + (rand() % 1000) / 100.0;
}

void SendPacketWithLoss(Ptr<Socket> socket, Ptr<Node> targetNode, double lossValue) {
    Ipv4Address targetAddress = targetNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    
    Ptr<Packet> packet = CreatePacketWithLoss(lossValue);
    socket->SendTo(packet, 0, InetSocketAddress(targetAddress, 9));
    
    NS_LOG_UNCOND("Sent packet with loss value " << lossValue << " to node " 
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
    
    if (!CheckGanEnvironment()) {
        NS_LOG_ERROR("GAN environment setup issues detected.");
        return 1;
    }
    
    if (verbose) {
        LogComponentEnable("DsrcHighwaySimple", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("Socket", LOG_LEVEL_INFO);
    }
    
    if (nVehicles < TOTAL_GAN_NODES + 5) {
        nVehicles = TOTAL_GAN_NODES + 5;
        NS_LOG_WARN("Increasing number of vehicles to " << nVehicles);
    }
    
    NS_LOG_INFO("Creating " << nVehicles << " vehicles");
    NodeContainer vehicles;
    vehicles.Create(nVehicles);
    
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
    
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    
    positionAlloc->Add(Vector(SIMULATION_AREA_SIZE * 0.25, SIMULATION_AREA_SIZE * 0.25, 0.0));
    positionAlloc->Add(Vector(SIMULATION_AREA_SIZE * 0.75, SIMULATION_AREA_SIZE * 0.25, 0.0));
    positionAlloc->Add(Vector(SIMULATION_AREA_SIZE * 0.25, SIMULATION_AREA_SIZE * 0.75, 0.0));
    positionAlloc->Add(Vector(SIMULATION_AREA_SIZE * 0.75, SIMULATION_AREA_SIZE * 0.75, 0.0));
    
    double radius = SIMULATION_AREA_SIZE * 0.1;
    for (uint32_t i = 0; i < 5; i++) {
        double angle = 2.0 * M_PI * i / 5.0;
        double x = SIMULATION_AREA_SIZE * 0.25 + radius * cos(angle);
        double y = SIMULATION_AREA_SIZE * 0.25 + radius * sin(angle);
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    for (uint32_t i = 0; i < 5; i++) {
        double angle = 2.0 * M_PI * i / 5.0;
        double x = SIMULATION_AREA_SIZE * 0.75 + radius * cos(angle);
        double y = SIMULATION_AREA_SIZE * 0.25 + radius * sin(angle);
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    for (uint32_t i = 0; i < 5; i++) {
        double angle = 2.0 * M_PI * i / 5.0;
        double x = SIMULATION_AREA_SIZE * 0.25 + radius * cos(angle);
        double y = SIMULATION_AREA_SIZE * 0.75 + radius * sin(angle);
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    for (uint32_t i = 0; i < 5; i++) {
        double angle = 2.0 * M_PI * i / 5.0;
        double x = SIMULATION_AREA_SIZE * 0.75 + radius * cos(angle);
        double y = SIMULATION_AREA_SIZE * 0.75 + radius * sin(angle);
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    
    for (uint32_t i = TOTAL_GAN_NODES; i < nVehicles; i++) {
        double x = SIMULATION_AREA_SIZE * ((double) rand() / RAND_MAX);
        double y = SIMULATION_AREA_SIZE * ((double) rand() / RAND_MAX);
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    
    MobilityHelper mobility;
    mobility.SetPositionAllocator(positionAlloc);
    
    for (uint32_t i = 0; i < NUM_GENERATORS; i++) {
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(vehicles.Get(i));
    }
    
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(
                                 SIMULATION_AREA_SIZE * 0.15, SIMULATION_AREA_SIZE * 0.35,
                                 SIMULATION_AREA_SIZE * 0.15, SIMULATION_AREA_SIZE * 0.35)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=2|Max=5]"));
    for (uint32_t i = NUM_GENERATORS; i < NUM_GENERATORS + 5; i++) {
        mobility.Install(vehicles.Get(i));
    }
    
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(
                                 SIMULATION_AREA_SIZE * 0.65, SIMULATION_AREA_SIZE * 0.85,
                                 SIMULATION_AREA_SIZE * 0.15, SIMULATION_AREA_SIZE * 0.35)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=2|Max=5]"));
    for (uint32_t i = NUM_GENERATORS + 5; i < NUM_GENERATORS + 10; i++) {
        mobility.Install(vehicles.Get(i));
    }
    
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(
                                 SIMULATION_AREA_SIZE * 0.15, SIMULATION_AREA_SIZE * 0.35,
                                 SIMULATION_AREA_SIZE * 0.65, SIMULATION_AREA_SIZE * 0.85)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=2|Max=5]"));
    for (uint32_t i = NUM_GENERATORS + 10; i < NUM_GENERATORS + 15; i++) {
        mobility.Install(vehicles.Get(i));
    }
    
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(
                                 SIMULATION_AREA_SIZE * 0.65, SIMULATION_AREA_SIZE * 0.85,
                                 SIMULATION_AREA_SIZE * 0.65, SIMULATION_AREA_SIZE * 0.85)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=2|Max=5]"));
    for (uint32_t i = NUM_GENERATORS + 15; i < TOTAL_GAN_NODES; i++) {
        mobility.Install(vehicles.Get(i));
    }
    
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(0, SIMULATION_AREA_SIZE, 0, SIMULATION_AREA_SIZE)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=10|Max=100]"));
    for (uint32_t i = TOTAL_GAN_NODES; i < nVehicles; i++) {
        mobility.Install(vehicles.Get(i));
    }
    
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
    
    std::vector<Ptr<Socket>> nodeSockets;
    uint16_t basePort = 9;
    
    for (uint32_t i = 0; i < TOTAL_GAN_NODES; i++) {
        TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
        Ptr<Socket> recvSocket = Socket::CreateSocket(vehicles.Get(i), tid);
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), basePort);
        if (recvSocket->Bind(local) == -1) {
            NS_LOG_ERROR("Failed to bind socket on node " << i);
            return -1;
        }
        recvSocket->SetRecvCallback(MakeCallback(&CustomServerReceiveCallback));
        NS_LOG_INFO("Socket bound on node " << i << " to port " << basePort);
        nodeSockets.push_back(recvSocket);
    }
    
    std::vector<ApplicationContainer> discToGenApps(NUM_GENERATORS);
    std::vector<ApplicationContainer> genToDiscApps(NUM_GENERATORS);
    ApplicationContainer genCommunicationApps;
    
    for (uint32_t gen = 0; gen < NUM_GENERATORS; gen++) {
        uint32_t startIdx = NUM_GENERATORS + gen * 5;
        uint32_t endIdx = startIdx + 5;
        
        for (uint32_t i = startIdx; i < endIdx; i++) {
            UdpEchoClientHelper client(interfaces.GetAddress(gen), basePort);
            client.SetAttribute("MaxPackets", UintegerValue(40));
            client.SetAttribute("Interval", TimeValue(Seconds(20.0)));  // Reduced interval
            client.SetAttribute("PacketSize", UintegerValue(512));
            
            ApplicationContainer app = client.Install(vehicles.Get(i));
            app.Start(Seconds(2.0 + (i - startIdx)));
            app.Stop(Seconds(simTime - 5.0));
            discToGenApps[gen].Add(app);
            
            uint32_t discId = MapToDiscriminatorId(i);
            double initialLoss = GenerateInitialLoss(i);
            lossTracker.updateDiscriminatorLoss(discId, initialLoss);
            
            Simulator::Schedule(Seconds(2.0 + (i - startIdx)), 
                              &SendPacketWithLoss, 
                              nodeSockets[i],
                              vehicles.Get(gen),
                              initialLoss);
            
            NS_LOG_INFO("Created client on discriminator " << i << " to send to generator " << gen + 1);
        }
    }
    
    for (uint32_t i = 0; i < NUM_GENERATORS; i++) {
        uint32_t genId = i + 1;
        double initialLoss = GenerateInitialLoss(i);
        lossTracker.updateGeneratorLoss(genId, initialLoss);
    }
    
    for (uint32_t gen = 0; gen < NUM_GENERATORS; gen++) {
        uint32_t startIdx = NUM_GENERATORS + gen * 5;
        uint32_t endIdx = startIdx + 5;
        
        for (uint32_t i = startIdx; i < endIdx; i++) {
            UdpEchoClientHelper client(interfaces.GetAddress(i), basePort);
            client.SetAttribute("MaxPackets", UintegerValue(40));
            client.SetAttribute("Interval", TimeValue(Seconds(20.0)));  // Reduced interval
            client.SetAttribute("PacketSize", UintegerValue(512));
            
            ApplicationContainer app = client.Install(vehicles.Get(gen));
            app.Start(Seconds(5.0 + (i - startIdx)));
            app.Stop(Seconds(simTime - 5.0));
            genToDiscApps[gen].Add(app);
            
            Simulator::Schedule(Seconds(5.0 + (i - startIdx)), 
                              &SendPacketWithLoss, 
                              nodeSockets[gen],
                              vehicles.Get(i),
                              lossTracker.getGeneratorLoss(gen + 1));
            
            NS_LOG_INFO("Created client on generator " << gen + 1 << " to send to discriminator " << i);
        }
    }
    
    for (uint32_t i = 0; i < NUM_GENERATORS; i++) {
        for (uint32_t j = 0; j < NUM_GENERATORS; j++) {
            if (i == j) continue;
            
            UdpEchoClientHelper client(interfaces.GetAddress(j), basePort);
            client.SetAttribute("MaxPackets", UintegerValue(30));
            client.SetAttribute("Interval", TimeValue(Seconds(20.0)));  // Reduced interval
            client.SetAttribute("PacketSize", UintegerValue(1024));
            
            ApplicationContainer app = client.Install(vehicles.Get(i));
            app.Start(Seconds(10.0 + i * 0.5));
            app.Stop(Seconds(simTime - 5.0));
            genCommunicationApps.Add(app);
            
            Simulator::Schedule(Seconds(10.0 + i * 0.5), 
                              &SendPacketWithLoss, 
                              nodeSockets[i],
                              vehicles.Get(j),
                              lossTracker.getGeneratorLoss(i + 1));
        }
    }
    
    for (uint32_t i = 0; i < TOTAL_GAN_NODES; i++) {
        for (double t = 15.0; t < simTime - 10.0; t += 20.0) {  // Reduced interval
            double delay = t + (rand() % 5);
            
            if (i < NUM_GENERATORS) {
                uint32_t genId = i + 1;
                
                Simulator::Schedule(Seconds(delay), 
                                   [i, genId, &vehicles, &nodeSockets]() {
                                       double currentLoss = lossTracker.getGeneratorLoss(genId);
                                       double improvementFactor = 0.9 + (rand() % 10) / 100.0;
                                       double newLoss = currentLoss * improvementFactor;
                                       
                                       bool improved = lossTracker.updateGeneratorLoss(genId, newLoss);
                                       
                                       if (improved) {
                                           NS_LOG_UNCOND("Scheduling improved loss " << newLoss << " from generator " << genId);
                                           LogLossUpdate("Generator", genId, newLoss, true);
                                           
                                           uint32_t startIdx = NUM_GENERATORS + (genId - 1) * 5;
                                           uint32_t endIdx = startIdx + 5;
                                           
                                           for (uint32_t j = startIdx; j < endIdx; j++) {
                                               SendPacketWithLoss(nodeSockets[i], vehicles.Get(j), newLoss);
                                           }
                                           
                                           for (uint32_t j = 0; j < NUM_GENERATORS; j++) {
                                               if (j != i) {
                                                   SendPacketWithLoss(nodeSockets[i], vehicles.Get(j), newLoss);
                                               }
                                           }
                                       }
                                   });
            } else {
                uint32_t discId = MapToDiscriminatorId(i);
                uint32_t genId = GetGeneratorForDiscriminator(i) + 1;
                uint32_t genNodeId = genId - 1;
                
                Simulator::Schedule(Seconds(delay), 
                                   [i, discId, genId, genNodeId, &vehicles, &nodeSockets]() {
                                       double currentLoss = lossTracker.getDiscriminatorLoss(discId);
                                       double improvementFactor = 0.9 + (rand() % 10) / 100.0;
                                       double newLoss = currentLoss * improvementFactor;
                                       
                                       bool improved = lossTracker.updateDiscriminatorLoss(discId, newLoss);
                                       
                                       if (improved) {
                                           NS_LOG_UNCOND("Scheduling improved loss " << newLoss << " from discriminator " << discId);
                                           LogLossUpdate("Discriminator", discId, newLoss, true);
                                           SendPacketWithLoss(nodeSockets[i], vehicles.Get(genNodeId), newLoss);
                                       }
                                   });
            }
        }
    }
    
    ApplicationContainer backgroundApps;
    for (uint32_t i = TOTAL_GAN_NODES; i < std::min(nVehicles, TOTAL_GAN_NODES + 15); i++) {
        uint32_t dest = (i + 1) % nVehicles;
        if (dest < TOTAL_GAN_NODES) dest = TOTAL_GAN_NODES;
        
        UdpEchoClientHelper client(interfaces.GetAddress(dest), basePort + 1);
        client.SetAttribute("MaxPackets", UintegerValue(10));
        client.SetAttribute("Interval", TimeValue(Seconds(120.0)));
        client.SetAttribute("PacketSize", UintegerValue(512));
        
        ApplicationContainer app = client.Install(vehicles.Get(i));
        app.Start(Seconds(10.0 + (i - TOTAL_GAN_NODES) * 0.2));
        app.Stop(Seconds(simTime - 10.0));
        backgroundApps.Add(app);
    }
    
    for (uint32_t i = 0; i < NUM_GENERATORS; i++) {
        std::ostringstream pcapName;
        pcapName << "dsrc-highway-gan-gen" << (i + 1);
        wifiPhy.EnablePcap(pcapName.str(), devices.Get(i));
    }
    
    for (uint32_t i = NUM_GENERATORS; i < TOTAL_GAN_NODES; i++) {
        std::ostringstream pcapName;
        pcapName << "dsrc-highway-gan-disc-" << (i - NUM_GENERATORS + 1);
        wifiPhy.EnablePcap(pcapName.str(), devices.Get(i));
    }
    
    AnimationInterface anim("dsrc-highway-gan-animation.xml");
    anim.SetMaxPktsPerTraceFile(100000);
    
    uint8_t genColors[][3] = {
        {255, 0, 0}, {255, 128, 0}, {255, 255, 0}, {128, 255, 0}
    };
    
    for (uint32_t i = 0; i < NUM_GENERATORS; i++) {
        anim.UpdateNodeColor(vehicles.Get(i), genColors[i][0], genColors[i][1], genColors[i][2]);
        anim.UpdateNodeSize(vehicles.Get(i), 15, 15);
        std::ostringstream desc;
        desc << "Generator " << (i + 1);
        anim.UpdateNodeDescription(vehicles.Get(i), desc.str());
    }
    
    uint8_t discColors[][5][3] = {
        {{0, 0, 255}, {0, 128, 255}, {0, 255, 255}, {0, 255, 128}, {0, 128, 128}},
        {{128, 0, 128}, {255, 0, 255}, {255, 0, 128}, {128, 0, 64}, {64, 0, 64}},
        {{0, 255, 0}, {128, 255, 128}, {64, 255, 64}, {0, 128, 0}, {0, 64, 0}},
        {{255, 128, 128}, {255, 192, 192}, {192, 128, 128}, {128, 64, 64}, {64, 32, 32}}
    };
    
    for (uint32_t gen = 0; gen < NUM_GENERATORS; gen++) {
        uint32_t startIdx = NUM_GENERATORS + gen * 5;
        for (uint32_t i = 0; i < 5; i++) {
            uint32_t nodeId = startIdx + i;
            anim.UpdateNodeColor(vehicles.Get(nodeId), 
                                discColors[gen][i][0], 
                                discColors[gen][i][1], 
                                discColors[gen][i][2]);
            anim.UpdateNodeSize(vehicles.Get(nodeId), 10, 10);
            std::ostringstream desc;
            desc << "Disc " << (gen * 5 + i + 1) << " (G" << (gen + 1) << ")";
            anim.UpdateNodeDescription(vehicles.Get(nodeId), desc.str());
        }
    }
    
    for (uint32_t i = TOTAL_GAN_NODES; i < nVehicles; i++) {
        anim.UpdateNodeColor(vehicles.Get(i), 180, 180, 180);
    }
    
    NS_LOG_INFO("Starting simulation for up to " << simTime << " seconds");
    Simulator::Stop(Seconds(simTime + 1));
    Simulator::Run();
    Simulator::Destroy();
    
    NS_LOG_INFO("Simulation completed");
    return 0;
}