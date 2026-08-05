#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/netanim-module.h"
#include "ns3/olsr-helper.h"  // Include OLSR routing

#include <cstdio>    // popen, fgets
#include <cstdlib>   // atoi
#include <sstream>   // ostringstream
#include <iostream>  // cout
#include <fstream>   // file operations
#include <map>       // for storing loss values
#include <string>    // string manipulation

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DsrcHighwaySimple");

// Constants for the new configuration
const uint32_t NUM_GENERATORS = 2;
const uint32_t NUM_DISCRIMINATORS = 10;
const uint32_t TOTAL_GAN_NODES = NUM_GENERATORS + NUM_DISCRIMINATORS;
const double SIMULATION_AREA_SIZE = 1000.0;

// Structure to track loss values for generators and discriminators
struct LossTracker {
    // Map to store current best loss values for each generator
    std::map<int, double> generatorLossValues;
    
    // Map to store current best loss values for each discriminator
    std::map<int, double> discriminatorLossValues;
    
    // Initialize with default high loss values (lower is better)
    LossTracker() {
        // Initialize generator loss values with high numbers
        for (int i = 1; i <= NUM_GENERATORS; i++) {
            generatorLossValues[i] = 9999.0;
        }
        
        // Initialize discriminator loss values with high numbers
        for (int i = 1; i <= NUM_DISCRIMINATORS; i++) {
            discriminatorLossValues[i] = 9999.0;
        }
    }
    
    // Update generator loss value if better
    bool updateGeneratorLoss(int genId, double newLoss) {
        if (newLoss < generatorLossValues[genId]) {
            generatorLossValues[genId] = newLoss;
            return true; // Loss was improved
        }
        return false; // No improvement
    }
    
    // Update discriminator loss value if better
    bool updateDiscriminatorLoss(int discId, double newLoss) {
        if (newLoss < discriminatorLossValues[discId]) {
            discriminatorLossValues[discId] = newLoss;
            return true; // Loss was improved
        }
        return false; // No improvement
    }
    
    // Get current best loss for a generator
    double getGeneratorLoss(int genId) {
        return generatorLossValues[genId];
    }
    
    // Get current best loss for a discriminator
    double getDiscriminatorLoss(int discId) {
        return discriminatorLossValues[discId];
    }
} lossTracker;

// Function to check if Python scripts exist and directories are ready
bool CheckGanEnvironment() {
    // Check for Python scripts
    std::string scriptPath1 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_generator1.py";
    std::string scriptPath2 = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_discriminator1.py";
    
    std::ifstream file1(scriptPath1.c_str());
    std::ifstream file2(scriptPath2.c_str());
    
    bool scriptsExist = file1.good() && file2.good();
    file1.close();
    file2.close();
    
    if (!scriptsExist) {
        NS_LOG_ERROR("GAN Python scripts not found at expected paths!");
        return false;
    }
    
    // Create necessary directories
    std::string baseDir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/";
    std::vector<std::string> dirs;

    dirs.push_back(baseDir + "results");
    dirs.push_back(baseDir + "results/generator_1");
    dirs.push_back(baseDir + "results/generator_2");
    
    for (int i = 1; i <= NUM_DISCRIMINATORS; i++) {
        std::ostringstream discDir;
        discDir << baseDir << "results/discriminator_" << i;
        dirs.push_back(discDir.str());
    }
    
    for (const std::string& dir : dirs) {
        std::string mkdirCmd = "mkdir -p " + dir;
        int status = system(mkdirCmd.c_str());
        if (status != 0) {
            NS_LOG_ERROR("Failed to create directory: " << dir);
            return false;
        }
    }
    
    // Create loss tracking file
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

// Function to map node ID to discriminator ID (for Python script)
int MapToDiscriminatorId(int nodeId) {
    // Nodes 2-11 map to discriminator IDs 1-10
    if (nodeId >= NUM_GENERATORS && nodeId < TOTAL_GAN_NODES) {
        return nodeId - NUM_GENERATORS + 1;
    }
    return -1; // Invalid
}

// Function to determine which generator a discriminator belongs to
int GetGeneratorForDiscriminator(int nodeId) {
    if (nodeId >= NUM_GENERATORS && nodeId < NUM_GENERATORS + 5) {
        return 0; // First 5 discriminators belong to generator 1
    } else if (nodeId >= NUM_GENERATORS + 5 && nodeId < TOTAL_GAN_NODES) {
        return 1; // Next 5 discriminators belong to generator 2
    }
    return -1; // Invalid
}

// Function to create a packet with loss information embedded
Ptr<Packet> CreatePacketWithLoss(double lossValue) {
    std::ostringstream oss;
    oss << lossValue;
    std::string lossStr = oss.str();
    
    // Create packet with loss value embedded
    uint8_t *buffer = new uint8_t[lossStr.length() + 1];
    memcpy(buffer, lossStr.c_str(), lossStr.length() + 1);
    
    Ptr<Packet> packet = Create<Packet>(buffer, lossStr.length() + 1);
    delete[] buffer;
    
    return packet;
}

// Function to extract loss value from packet
double ExtractLossFromPacket(Ptr<Packet> packet) {
    uint8_t *buffer = new uint8_t[packet->GetSize()];
    packet->CopyData(buffer, packet->GetSize());
    
    // Convert buffer to string and parse double
    std::string lossStr((char*)buffer, packet->GetSize());
    delete[] buffer;
    
    // Try to parse the loss value
    try {
        double lossValue = std::stod(lossStr);
        return lossValue;
    } catch (const std::exception& e) {
        NS_LOG_ERROR("Failed to parse loss value from packet: " << lossStr);
        return 9999.0; // Default high loss value
    }
}

// Function to log loss updates to file
void LogLossUpdate(const std::string& nodeType, int nodeId, double lossValue, bool improved) {
    std::string baseDir = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/";
    std::string lossFilePath = baseDir + "results/loss_tracking.csv";
    
    std::ofstream lossFile(lossFilePath.c_str(), std::ios::app);
    if (lossFile.is_open()) {
        time_t now = time(NULL);
        lossFile << now << "," << nodeType << "," << nodeId << "," << lossValue << "," << (improved ? "Yes" : "No") << "\n";
        lossFile.close();
    } else {
        NS_LOG_ERROR("Failed to open loss tracking file for writing");
    }
}

// Custom Receive Callback with improved error handling, logging, and loss tracking
void CustomServerReceiveCallback(Ptr<Socket> socket) {
    if (!socket) {
        NS_LOG_ERROR("Invalid socket pointer");
        return;
    }
    NS_LOG_INFO("CustomServerReceiveCallback triggered on node " << socket->GetNode()->GetId());
    printf("*************#*************\n");

    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from))) {
        uint32_t nodeId = socket->GetNode()->GetId();
        InetSocketAddress inetFrom = InetSocketAddress::ConvertFrom(from);
        Ipv4Address senderAddr = inetFrom.GetIpv4();

        // Convert Ipv4Address to string using ostringstream
        std::ostringstream addrStream;
        senderAddr.Print(addrStream);
        std::string senderAddrStr = addrStream.str();

        NS_LOG_INFO("Node " << nodeId << " received packet from " << senderAddr << " of size " << packet->GetSize());
        printf("Received packet on node %u from %s\n", nodeId, senderAddrStr.c_str());

        // Extract loss value from the packet
        double receivedLoss = ExtractLossFromPacket(packet);
        NS_LOG_UNCOND("Extracted loss value: " << receivedLoss);

        // Only process packets for GAN nodes (0-11: 2 generators + 10 discriminators)
        if (nodeId >= TOTAL_GAN_NODES) {
            NS_LOG_UNCOND("Node " << nodeId << " is not configured for GAN training");
            continue;
        }

        // Handle loss value based on node type
        bool improvedLoss = false;
        double currentBestLoss = 9999.0;
        
        // Nodes 0-1 are generators receiving from discriminators
        // Nodes 2-11 are discriminators receiving from generators
        if (nodeId < NUM_GENERATORS) {
            int genId = nodeId + 1; // Convert to 1-based generator ID
            
            // When generator receives from discriminator, update its loss if better
            improvedLoss = lossTracker.updateGeneratorLoss(genId, receivedLoss);
            currentBestLoss = lossTracker.getGeneratorLoss(genId);
            
            // Log the loss update
            LogLossUpdate("Generator", genId, receivedLoss, improvedLoss);
            
            if (improvedLoss) {
                NS_LOG_UNCOND("Generator " << genId << " loss improved to " << currentBestLoss);
            } else {
                NS_LOG_UNCOND("Generator " << genId << " received loss " << receivedLoss 
                            << " (no improvement from current best: " << currentBestLoss << ")");
            }
            
        } else if (nodeId >= NUM_GENERATORS && nodeId < TOTAL_GAN_NODES) {
            int discId = MapToDiscriminatorId(nodeId);
            int genId = GetGeneratorForDiscriminator(nodeId) + 1; // Convert to 1-based generator ID
            
            // When discriminator receives from generator, update its loss if better
            improvedLoss = lossTracker.updateDiscriminatorLoss(discId, receivedLoss);
            currentBestLoss = lossTracker.getDiscriminatorLoss(discId);
            
            // Log the loss update
            LogLossUpdate("Discriminator", discId, receivedLoss, improvedLoss);
            
            if (improvedLoss) {
                NS_LOG_UNCOND("Discriminator " << discId << " loss improved to " << currentBestLoss);
            } else {
                NS_LOG_UNCOND("Discriminator " << discId << " received loss " << receivedLoss 
                            << " (no improvement from current best: " << currentBestLoss << ")");
            }
        }

        // Create a command with full path
        std::ostringstream cmd;

        // Nodes 0-1 are generators
        // Nodes 2-11 are discriminators
        if (nodeId < NUM_GENERATORS) {
            int genId = nodeId + 1; // Convert to 1-based generator ID
            printf("**************************\n");
            cmd << "python3 /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_generator1.py " 
                << genId << " " << senderAddr.Get() << " " << currentBestLoss
                << " > /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/generator_" 
                << genId << "/log_" << time(NULL) << ".txt 2>&1";
                
            NS_LOG_UNCOND("Running generator " << genId << " with command: " << cmd.str());
        } else if (nodeId >= NUM_GENERATORS && nodeId < TOTAL_GAN_NODES) {
            int discId = MapToDiscriminatorId(nodeId);
            int genId = GetGeneratorForDiscriminator(nodeId) + 1; // Convert to 1-based generator ID
            
            cmd << "python3 /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_discriminator1.py " 
                << discId << " " << senderAddr.Get() << " " << genId << " " << currentBestLoss
                << " > /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/results/discriminator_" 
                << discId << "/log_" << time(NULL) << ".txt 2>&1";
                
            NS_LOG_UNCOND("Running discriminator " << discId << " for generator " 
                        << genId << " with command: " << cmd.str());
        } else {
            NS_LOG_UNCOND("Node " << nodeId << " is not configured for GAN training");
            continue;
        }

        NS_LOG_UNCOND("Executing: " << cmd.str());

        // Run command in background to avoid blocking simulation
        std::string bgCmd = cmd.str() + " &";

        // Check if the Python script exists before running
        std::string scriptPath = (nodeId < NUM_GENERATORS) ? 
            "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_generator1.py" : 
            "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/run_discriminator1.py";

        std::ifstream scriptFile(scriptPath.c_str());
        if (!scriptFile.good()) {
            NS_LOG_ERROR("Python script not found or not accessible: " << scriptPath);
            continue;
        }
        scriptFile.close();

        try {
            NS_LOG_UNCOND("About to execute system command");
            int status = system(bgCmd.c_str());
            NS_LOG_UNCOND("Command executed with status: " << status);

            if (status != 0) {
                NS_LOG_ERROR("Failed to start GAN script with status: " << status);
            } else {
                NS_LOG_UNCOND("GAN script started in background");
            }
        } catch (const std::exception& e) {
            NS_LOG_ERROR("Exception during command execution: " << e.what());
        }
    }
}

// Function to simulate initial loss values from GAN training
double GenerateInitialLoss(int nodeId) {
    // Randomize initial loss values based on node ID for simulation
    srand(time(NULL) + nodeId);
    double baseLoss = 2.0 + (rand() % 1000) / 100.0; // Between 2.0 and 12.0
    return baseLoss;
}

// Send custom packet with loss value
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
    double simTime = 150.0; // Even longer simulation time for GAN training
    bool verbose = true;
    printf("Hi Ferry\n");
    
    CommandLine cmd(__FILE__);
    cmd.AddValue("nVehicles", "Number of vehicles", nVehicles);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("verbose", "Enable verbose logging", verbose);
    cmd.Parse(argc, argv);
    printf("Hiiii Again \n");
    
    // Check GAN environment at the beginning
    if (!CheckGanEnvironment()) {
        NS_LOG_ERROR("GAN environment setup issues detected. Please check the paths and permissions.");
        return 1;
    }

    // Enable logging
    if (verbose)
    {
        LogComponentEnable("DsrcHighwaySimple", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("Socket", LOG_LEVEL_INFO);
    }
    printf("Hellllo Again \n");
    
    // Make sure we have enough vehicles for all GAN nodes plus regular traffic
    if (nVehicles < TOTAL_GAN_NODES + 5) {
        nVehicles = TOTAL_GAN_NODES + 5;
        NS_LOG_WARN("Increasing number of vehicles to " << nVehicles << " to accommodate all GAN nodes");
    }
    
    // Create vehicle nodes
    NS_LOG_INFO("Creating " << nVehicles << " vehicles");
    NodeContainer vehicles;
    vehicles.Create(nVehicles);

    // WiFi 802.11p setup
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);

    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent", DoubleValue(3.0));

    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(23));
    wifiPhy.Set("TxPowerEnd", DoubleValue(23));
    wifiPhy.Set("TxGain", DoubleValue(5)); // Increase transmit power gain
    wifiPhy.Set("RxGain", DoubleValue(5)); // Increase receive gain

    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue("OfdmRate6MbpsBW10MHz"),
                               "ControlMode", StringValue("OfdmRate6MbpsBW10MHz"));

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, vehicles);

    // Strategic positioning of GAN nodes
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    
    // Position generators - one at center, one at edge
    positionAlloc->Add(Vector(SIMULATION_AREA_SIZE * 0.5, SIMULATION_AREA_SIZE * 0.5, 0.0));  // Generator 1 at center
    positionAlloc->Add(Vector(SIMULATION_AREA_SIZE * 0.75, SIMULATION_AREA_SIZE * 0.75, 0.0)); // Generator 2 at quadrant
    
    // Position discriminators in two circles around each generator
    double radius1 = SIMULATION_AREA_SIZE * 0.15; // Radius for first generator's circle
    double radius2 = SIMULATION_AREA_SIZE * 0.1;  // Radius for second generator's circle
    
    // 5 discriminators around first generator (nodes 2-6)
    for (uint32_t i = 0; i < 5; i++) {
        double angle = 2.0 * M_PI * i / 5.0;
        double x = SIMULATION_AREA_SIZE * 0.5 + radius1 * cos(angle);
        double y = SIMULATION_AREA_SIZE * 0.5 + radius1 * sin(angle);
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    
    // 5 discriminators around second generator (nodes 7-11)
    for (uint32_t i = 0; i < 5; i++) {
        double angle = 2.0 * M_PI * i / 5.0;
        double x = SIMULATION_AREA_SIZE * 0.75 + radius2 * cos(angle);
        double y = SIMULATION_AREA_SIZE * 0.75 + radius2 * sin(angle);
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    
    // Random positions for other nodes across the 1000x1000 area
    for (uint32_t i = TOTAL_GAN_NODES; i < nVehicles; i++) {
        double x = SIMULATION_AREA_SIZE * ((double) rand() / RAND_MAX);
        double y = SIMULATION_AREA_SIZE * ((double) rand() / RAND_MAX);
        positionAlloc->Add(Vector(x, y, 0.0));
    }

    // Different mobility models for different node types
    MobilityHelper mobility;
    mobility.SetPositionAllocator(positionAlloc);
    
    // Generators stay fixed
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(vehicles.Get(0));
    mobility.Install(vehicles.Get(1));
    
    // Discriminators move slowly within bounds around their respective generators
    // First group (around generator 1)
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(
                                 SIMULATION_AREA_SIZE * 0.3, 
                                 SIMULATION_AREA_SIZE * 0.7, 
                                 SIMULATION_AREA_SIZE * 0.3, 
                                 SIMULATION_AREA_SIZE * 0.7)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=2|Max=5]"));
    
    for (uint32_t i = NUM_GENERATORS; i < NUM_GENERATORS + 5; i++) {
        mobility.Install(vehicles.Get(i));
    }
    
    // Second group (around generator 2)
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(
                                 SIMULATION_AREA_SIZE * 0.6, 
                                 SIMULATION_AREA_SIZE * 0.9, 
                                 SIMULATION_AREA_SIZE * 0.6, 
                                 SIMULATION_AREA_SIZE * 0.9)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=2|Max=5]"));
    
    for (uint32_t i = NUM_GENERATORS + 5; i < TOTAL_GAN_NODES; i++) {
        mobility.Install(vehicles.Get(i));
    }
    
    // Other vehicles use normal mobility across the full area
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue(Rectangle(0, SIMULATION_AREA_SIZE, 0, SIMULATION_AREA_SIZE)),
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=10|Max=100]"));
    
    for (uint32_t i = TOTAL_GAN_NODES; i < nVehicles; i++) {
        mobility.Install(vehicles.Get(i));
    }

    // Internet stack setup with OLSR routing
    InternetStackHelper internet;
    OlsrHelper olsr;  // Add OLSR routing to ensure packet delivery
    internet.SetRoutingHelper(olsr);
    internet.Install(vehicles);

    // IP addressing
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4h.Assign(devices);
    
    // Configure nodes for IP forwarding
    for (uint32_t i = 0; i < nVehicles; i++) {
        Ptr<Ipv4> ipv4Interface = vehicles.Get(i)->GetObject<Ipv4>();
        ipv4Interface->SetAttribute("IpForward", BooleanValue(true));
    }

    // Create custom sockets for GAN nodes (generators and discriminators)
    std::vector<Ptr<Socket>> nodeSockets;
    uint16_t basePort = 9;
    
    for (uint32_t i = 0; i < TOTAL_GAN_NODES; i++) {
        TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
        Ptr<Socket> recvSocket = Socket::CreateSocket(vehicles.Get(i), tid);
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), basePort); // Bind to any address on port 9
        if (recvSocket->Bind(local) == -1) {
            NS_LOG_ERROR("Failed to bind socket on node " << i);
            return -1;
        }
        recvSocket->SetRecvCallback(MakeCallback(&CustomServerReceiveCallback));
        NS_LOG_INFO("Socket bound on node " << i << " to port " << basePort);
        nodeSockets.push_back(recvSocket);
    }

    // Create all application containers
    ApplicationContainer discToGen1Apps;
    ApplicationContainer discToGen2Apps;
    ApplicationContainer gen1ToDiscApps;
    ApplicationContainer gen2ToDiscApps;
    ApplicationContainer genCommunicationApps;
    
    // INCREASED COMMUNICATION FREQUENCY AND PACKETS WITH CUSTOM LOSS VALUES
    
    // 1. First set of discriminators (nodes 2-6) send to generator 1 (node 0)
    for (uint32_t i = NUM_GENERATORS; i < NUM_GENERATORS + 5; i++) {
        UdpEchoClientHelper client(interfaces.GetAddress(0), basePort);
        client.SetAttribute("MaxPackets", UintegerValue(40));       // More packets
        client.SetAttribute("Interval", TimeValue(Seconds(40.0)));  // More frequent communication
        client.SetAttribute("PacketSize", UintegerValue(1024));
        
        ApplicationContainer app = client.Install(vehicles.Get(i));
        app.Start(Seconds(5.0 + (i - NUM_GENERATORS))); // Staggered start
        app.Stop(Seconds(simTime - 5.0));
        discToGen1Apps.Add(app);
        
        // Initialize with random loss value and schedule future loss updates
        int discId = MapToDiscriminatorId(i);
        double initialLoss = GenerateInitialLoss(i);
        lossTracker.updateDiscriminatorLoss(discId, initialLoss);
        
        // Schedule initial packet with loss value
        Simulator::Schedule(Seconds(5.0 + (i - NUM_GENERATORS)), 
                           &SendPacketWithLoss, 
                           nodeSockets[i],
                           vehicles.Get(0),
                           initialLoss);
        
        NS_LOG_INFO("Created client on discriminator " << i << " to send to generator 1 at " << interfaces.GetAddress(0));
    }
    
    // 2. Second set of discriminators (nodes 7-11) send to generator 2 (node 1)
    for (uint32_t i = NUM_GENERATORS + 5; i < TOTAL_GAN_NODES; i++) {
        UdpEchoClientHelper client(interfaces.GetAddress(1), basePort);
        client.SetAttribute("MaxPackets", UintegerValue(40));       // More packets
        client.SetAttribute("Interval", TimeValue(Seconds(40.0)));  // More frequent communication
        client.SetAttribute("PacketSize", UintegerValue(1024));
        
        ApplicationContainer app = client.Install(vehicles.Get(i));
        app.Start(Seconds(5.0 + (i - NUM_GENERATORS - 5))); // Staggered start
        app.Stop(Seconds(simTime - 5.0));
        discToGen2Apps.Add(app);
        
        // Initialize with random loss value and schedule future loss updates
        int discId = MapToDiscriminatorId(i);
        double initialLoss = GenerateInitialLoss(i);
        lossTracker.updateDiscriminatorLoss(discId, initialLoss);
        
        // Schedule initial packet with loss value
        Simulator::Schedule(Seconds(5.0 + (i - NUM_GENERATORS - 5)), 
                           &SendPacketWithLoss, 
                           nodeSockets[i],
                           vehicles.Get(1),
                           initialLoss);
        
        NS_LOG_INFO("Created client on discriminator " << i << " to send to generator 2 at " << interfaces.GetAddress(1));
    }
    
    // Initialize generator loss values
    for (uint32_t i = 0; i < NUM_GENERATORS; i++) {
        int genId = i + 1;
        double initialLoss = GenerateInitialLoss(i);
        lossTracker.updateGeneratorLoss(genId, initialLoss);
    }
    
    // 3. Generator 1 sends to its discriminators (nodes 2-6)
    for (uint32_t i = NUM_GENERATORS; i < NUM_GENERATORS + 5; i++) {
        UdpEchoClientHelper client(interfaces.GetAddress(i), basePort);
        client.SetAttribute("MaxPackets", UintegerValue(40));       // More packets
        client.SetAttribute("Interval", TimeValue(Seconds(40.0)));  // More frequent communication
        client.SetAttribute("PacketSize", UintegerValue(1024));
        
        ApplicationContainer app = client.Install(vehicles.Get(0));
        app.Start(Seconds(10.0 + (i - NUM_GENERATORS))); // Start after discriminators have sent
        app.Stop(Seconds(simTime - 5.0));
        gen1ToDiscApps.Add(app);
        
        // Schedule initial packet with loss value
        Simulator::Schedule(Seconds(10.0 + (i - NUM_GENERATORS)), 
                           &SendPacketWithLoss, 
                           nodeSockets[0],
                           vehicles.Get(i),
                           lossTracker.getGeneratorLoss(1));
        
        NS_LOG_INFO("Created client on generator 1 to send to discriminator " << i << " at " << interfaces.GetAddress(i));
    }
    
    // 4. Generator 2 sends to its discriminators (nodes 7-11)
    for (uint32_t i = NUM_GENERATORS + 5; i < TOTAL_GAN_NODES; i++) {
        UdpEchoClientHelper client(interfaces.GetAddress(i), basePort);
        client.SetAttribute("MaxPackets", UintegerValue(40));       // More packets
        client.SetAttribute("Interval", TimeValue(Seconds(40.0)));  // More frequent communication
        client.SetAttribute("PacketSize", UintegerValue(1024));
        
        ApplicationContainer app = client.Install(vehicles.Get(1));
        app.Start(Seconds(10.0 + (i - NUM_GENERATORS - 5))); // Start after discriminators have sent
        app.Stop(Seconds(simTime - 5.0));
        gen2ToDiscApps.Add(app);
        
        // Schedule initial packet with loss value
        Simulator::Schedule(Seconds(10.0 + (i - NUM_GENERATORS - 5)), 
                           &SendPacketWithLoss, 
                           nodeSockets[1],
                           vehicles.Get(i),
                           lossTracker.getGeneratorLoss(2));
        
        NS_LOG_INFO("Created client on generator 2 to send to discriminator " << i << " at " << interfaces.GetAddress(i));
    }
    
    // 5. Generator 1 and Generator 2 communicate with each other
    // Generator 1 sends to Generator 2
    UdpEchoClientHelper gen1ToGen2(interfaces.GetAddress(1), basePort);
    gen1ToGen2.SetAttribute("MaxPackets", UintegerValue(30));      // More packets
    gen1ToGen2.SetAttribute("Interval", TimeValue(Seconds(40.0))); // More frequent communication
    gen1ToGen2.SetAttribute("PacketSize", UintegerValue(2048));
    
    ApplicationContainer gen1ToGen2App = gen1ToGen2.Install(vehicles.Get(0));
    gen1ToGen2App.Start(Seconds(15.0));
    gen1ToGen2App.Stop(Seconds(simTime - 5.0));
    genCommunicationApps.Add(gen1ToGen2App);
    
    // Schedule initial packet with loss value between generators
    Simulator::Schedule(Seconds(15.0), 
                       &SendPacketWithLoss, 
                       nodeSockets[0],
                       vehicles.Get(1),
                       lossTracker.getGeneratorLoss(1));
    
    // Generator 2 sends to Generator 1
    UdpEchoClientHelper gen2ToGen1(interfaces.GetAddress(0), basePort);
    gen2ToGen1.SetAttribute("MaxPackets", UintegerValue(30));      // More packets
    gen2ToGen1.SetAttribute("Interval", TimeValue(Seconds(40.0))); // More frequent communication
    gen2ToGen1.SetAttribute("PacketSize", UintegerValue(2048));
    
    ApplicationContainer gen2ToGen1App = gen2ToGen1.Install(vehicles.Get(1));
    gen2ToGen1App.Start(Seconds(20.0));
    gen2ToGen1App.Stop(Seconds(simTime - 5.0));
    genCommunicationApps.Add(gen2ToGen1App);
    
    // Schedule initial packet with loss value between generators
    Simulator::Schedule(Seconds(20.0), 
                       &SendPacketWithLoss, 
                       nodeSockets[1],
                       vehicles.Get(0),
                       lossTracker.getGeneratorLoss(2));
    
    // Schedule periodic loss value updates (simulating GAN training progress)
    for (uint32_t i = 0; i < TOTAL_GAN_NODES; i++) {
        // Skip non-GAN nodes
        if (i >= TOTAL_GAN_NODES) continue;
        
        // For each node, schedule multiple loss updates throughout the simulation
        for (double t = 30.0; t < simTime - 10.0; t += 40.0) {
            // Add small random delay to avoid synchronization
            double delay = t + (rand() % 10);
            
            // For generators (nodes 0-1)
            if (i < NUM_GENERATORS) {
                int genId = i + 1;
                
                // Schedule loss updates to all connected discriminators
                Simulator::Schedule(Seconds(delay), 
                                   [i, genId, &vehicles, &nodeSockets]() {
                                       // Get current best loss
                                       double currentLoss = lossTracker.getGeneratorLoss(genId);
                                       
                                       // Generate slightly improved loss (90-99% of previous)
                                       double improvementFactor = 0.9 + (rand() % 10) / 100.0;
                                       double newLoss = currentLoss * improvementFactor;
                                       
                                       // Update loss tracker if better
                                       bool improved = lossTracker.updateGeneratorLoss(genId, newLoss);
                                       
                                       if (improved) {
                                           NS_LOG_UNCOND("Scheduling improved loss " << newLoss << " from generator " << genId);
                                           
                                           // Log the loss update
                                           LogLossUpdate("Generator", genId, newLoss, true);
                                           
                                           // Send to all discriminators
                                           int startIdx = (genId == 1) ? NUM_GENERATORS : NUM_GENERATORS + 5;
                                           int endIdx = (genId == 1) ? NUM_GENERATORS + 5 : TOTAL_GAN_NODES;
                                           
                                           for (int j = startIdx; j < endIdx; j++) {
                                               SendPacketWithLoss(nodeSockets[i], vehicles.Get(j), newLoss);
                                           }
                                           
                                           // Send to other generator
                                           int otherGenIdx = (i == 0) ? 1 : 0;
                                           SendPacketWithLoss(nodeSockets[i], vehicles.Get(otherGenIdx), newLoss);
                                       }
                                   });
            }
            // For discriminators (nodes 2-11)
            else {
                int discId = MapToDiscriminatorId(i);
                int genId = GetGeneratorForDiscriminator(i) + 1;
                int genNodeId = genId - 1;
                
                // Schedule loss updates to connected generator
                Simulator::Schedule(Seconds(delay), 
                                   [i, discId, genId, genNodeId, &vehicles, &nodeSockets]() {
                                       // Get current best loss
                                       double currentLoss = lossTracker.getDiscriminatorLoss(discId);
                                       
                                       // Generate slightly improved loss (90-99% of previous)
                                       double improvementFactor = 0.9 + (rand() % 10) / 100.0;
                                       double newLoss = currentLoss * improvementFactor;
                                       
                                       // Update loss tracker if better
                                       bool improved = lossTracker.updateDiscriminatorLoss(discId, newLoss);
                                       
                                       if (improved) {
                                           NS_LOG_UNCOND("Scheduling improved loss " << newLoss << " from discriminator " << discId);
                                           
                                           // Log the loss update
                                           LogLossUpdate("Discriminator", discId, newLoss, true);
                                           
                                           // Send to generator
                                           SendPacketWithLoss(nodeSockets[i], vehicles.Get(genNodeId), newLoss);
                                       }
                                   });
            }
        }
    }
    
    // Add simple background traffic with increased activity
    ApplicationContainer backgroundApps;
    for (uint32_t i = TOTAL_GAN_NODES; i < std::min(nVehicles, TOTAL_GAN_NODES + 15); i++) {
        uint32_t dest = (i + 1) % nVehicles;
        if (dest < TOTAL_GAN_NODES) dest = TOTAL_GAN_NODES; // Don't send to GAN nodes
        
        UdpEchoClientHelper client(interfaces.GetAddress(dest), basePort + 1); // Use different port
        client.SetAttribute("MaxPackets", UintegerValue(10));
        client.SetAttribute("Interval", TimeValue(Seconds(120.0)));
        client.SetAttribute("PacketSize", UintegerValue(512));
        
        ApplicationContainer app = client.Install(vehicles.Get(i));
        app.Start(Seconds(10.0 + (i - TOTAL_GAN_NODES) * 0.2));
        app.Stop(Seconds(simTime - 10.0));
        backgroundApps.Add(app);
    }

    // Enable PCAP tracing for all GAN nodes
    // Generator 1
    wifiPhy.EnablePcap("dsrc-highway-gan-gen1", devices.Get(0));
    // Generator 2
    wifiPhy.EnablePcap("dsrc-highway-gan-gen2", devices.Get(1));
    
    // Discriminators for both generators
    for (uint32_t i = NUM_GENERATORS; i < TOTAL_GAN_NODES; i++) {
        std::ostringstream pcapName;
        pcapName << "dsrc-highway-gan-disc-" << (i - NUM_GENERATORS + 1);
        wifiPhy.EnablePcap(pcapName.str(), devices.Get(i));
    }

    // NetAnim setup with improved visuals
    AnimationInterface anim("dsrc-highway-gan-animation.xml");
    anim.SetMaxPktsPerTraceFile(100000); // Increase packet count for animation
    
    // Set node colors for generators
    anim.UpdateNodeColor(vehicles.Get(0), 255, 0, 0);  // Red for generator 1
    anim.UpdateNodeSize(vehicles.Get(0), 15, 15);      // Larger size
    
    anim.UpdateNodeColor(vehicles.Get(1), 255, 128, 0);  // Orange for generator 2
    anim.UpdateNodeSize(vehicles.Get(1), 15, 15);        // Larger size
    
    // Different colors for first set of discriminators (nodes 2-6)
    uint8_t discColors1[][3] = {
        {0, 0, 255},     // Blue
        {0, 128, 255},   // Light Blue
        {0, 255, 255},   // Cyan
        {0, 255, 128},   // Teal
        {0, 128, 128}    // Dark Teal
    };
    
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t nodeId = i + NUM_GENERATORS;
        anim.UpdateNodeColor(vehicles.Get(nodeId), 
                            discColors1[i][0], 
                            discColors1[i][1], 
                            discColors1[i][2]);
        anim.UpdateNodeSize(vehicles.Get(nodeId), 10, 10);
        
        std::ostringstream desc;
        desc << "Disc " << (i+1) << " (G1)";
        anim.UpdateNodeDescription(vehicles.Get(nodeId), desc.str());
    }
    
    // Different colors for second set of discriminators (nodes 7-11)
    uint8_t discColors2[][3] = {
        {128, 0, 128},   // Purple
        {255, 0, 255},   // Magenta
        {255, 0, 128},   // Pink
        {128, 0, 64},    // Maroon
        {64, 0, 64}      // Dark Purple
    };
    
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t nodeId = i + NUM_GENERATORS + 5;
        anim.UpdateNodeColor(vehicles.Get(nodeId), 
                            discColors2[i][0], 
                            discColors2[i][1], 
                            discColors2[i][2]);
        anim.UpdateNodeSize(vehicles.Get(nodeId), 10, 10);
        
        std::ostringstream desc;
        desc << "Disc " << (i+6) << " (G2)";
        anim.UpdateNodeDescription(vehicles.Get(nodeId), desc.str());
    }
    
    // Set regular nodes to gray
    for (uint32_t i = TOTAL_GAN_NODES; i < nVehicles; i++) {
        anim.UpdateNodeColor(vehicles.Get(i), 180, 180, 180);  // Gray
    }

    // Add descriptive labels for generators
    anim.UpdateNodeDescription(vehicles.Get(0), "Generator 1");
    anim.UpdateNodeDescription(vehicles.Get(1), "Generator 2");
    
    // Run simulation
    NS_LOG_INFO("Starting simulation for " << simTime << " seconds");
    Simulator::Stop(Seconds(simTime + 1));
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_INFO("Simulation completed");
    return 0;
} //////