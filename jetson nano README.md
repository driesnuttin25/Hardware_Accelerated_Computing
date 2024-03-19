# Connecting the Jetson Nano to the Internet & Basic Handling
## Introduction
This is a basic guide on how to handle the Jetson Nano for further projects.

## Prerequisites
#### List of materials needed 
- Ethernet cable connected to laptop
- HDMI cable connected to output screen
- If needed mouse and keyboard connected to usb inputs

## Hardware Setup
- Username
```
team3
```
- Password
```
123456789
```
- Name of device
```
jetsonNanoTeam3
```

## Ethernet Connection on the jetson nano
Because the jetson nano does not have an onboard internet card it is neccesary to connect it via ethernet cable. 
On your main computer you can do the following: 

### Main computer
You should see the following connections in your "Network Connections" window on your computer

<img width="200" alt="image" src="https://github.com/driesnuttin25/Hardware_Accelerated_Computing/assets/114076101/9969e2ab-8042-401e-9ac1-bce27e4fa262">

In your wifi proporties, click sharing to enable the "Internet Connection Sharing"

<img width="268" alt="image" src="https://github.com/driesnuttin25/Hardware_Accelerated_Computing/assets/114076101/35ae23ba-87ea-4223-91d5-4567cccfc6df">

Open your "Command prompt" and type in the following command
```
ipconfig
```
Look for the following:
- IPv4 adress
- Subnet Mask
- Default gateway

for me this will be:
```
192.168.137.1
255.255.255.0
190.168.137.1
```
### On the jetson nano
To enable the sharing of internet please follow the next steps
- Proceed to the "Edit connections"
- Under "Ethernet" there should be a connection
- Proceed here and go to IPv4 settings
- Set the Method to "Manual"
- Add a new adress and fill in the adress (change the last digit) and the netmask and gateway
```
192.168.137.10 
255.255.255.0
190.168.137.1
```
On your Jetson nano terminal you can try the following command
```
ifconfig
```
To see if your internet connection is active.

To test your internet you can launch the chrome module or you can go to the terminal and try the following command
```
ping www.google.com
```
