#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//#include <MeOrion.h>
#include <SoftwareSerial.h>
//#include <Wire.h>
#include <AccelStepper.h>

#define pumpPin A5
#define lightPin A4

#define dirPinTheta 2
#define stepPinTheta 3
#define dirPinZ 4
#define stepPinZ 5
#define enablePinTheta 7
#define enablePinZ 6
#define dirPinX 10
#define stepPinX 11
#define dirPinY 8
#define stepPinY 9
#define enablePinY 13
#define enablePinX 12
#define stepsPerRevolution 200



#define delay_time 200

//int V_y = 150;
const int V_tot = 300;
const int A_tot = 2000;

int newData = false;

const byte numChars = 10000;
char receivedChars[numChars];   // an array to store the received data


//int dirPin = mePort[PORT_1].s1;
//int stpPin = mePort[PORT_1].s2;

AccelStepper stepperX(1, stepPinX, dirPinX);
AccelStepper stepperY(1, stepPinY, dirPinY);
AccelStepper stepperTheta(1, stepPinTheta, dirPinTheta);
AccelStepper stepperZ(1, stepPinZ, dirPinZ);

typedef struct command {
  int x_steps;
  int x_speed;
  int x_acc;
  int y_steps;
  int y_speed;
  int y_acc;
  int theta_steps;
  int theta_speed;
  int theta_acc;
  int z_steps;
  int z_speed;
  int z_acc;
  int pump_on;
  int l_on;
} command;

void setup() {
//  Serial.begin(19200);
  Serial.begin(9600);
  Serial.println("<Arduino is ready>");

  for(int i = 0; i < 14; i++) {
    pinMode(i, OUTPUT);
  }
  pinMode(pumpPin, OUTPUT);
  pinMode(lightPin, OUTPUT);

  // disable all other devices
  digitalWrite(enablePinX, HIGH);
  digitalWrite(enablePinY, HIGH);
  digitalWrite(enablePinZ, HIGH);
  digitalWrite(enablePinTheta, HIGH);
  digitalWrite(pumpPin, HIGH);
  // light pins?


  //CHECK ALL NUMBERS AND ADD PUMP/LIGHTS

  stepperX.setMaxSpeed(2000);
  stepperX.setAcceleration(5000);

  stepperY.setMaxSpeed(1000);
  stepperY.setAcceleration(2500);

  stepperTheta.setMaxSpeed(250);
  stepperTheta.setAcceleration(1000);

  stepperZ.setMaxSpeed(1000);
  stepperZ.setAcceleration(A_tot);

}

void loop() {
  // put your main code here, to run repeatedly:
//  String cmnd_string = "";
//  char arr[200];
//  int i = 0;
 
//  while (!Serial.available());
//  arr[i] = Serial.read();
//  cmnd_string += arr[i];
//  i++;
//  while (arr[i-1] != '|') {
//    while (!Serial.available());
//    arr[i] = Serial.read();
//    cmnd_string += arr[i];
//    i++;
//  }
//
//  arr[i] = 0;

//  Serial.println(cmnd_string);

  process_string();
 

//  while (Serial.available()) {
//    char enter = Serial.read();
//  }
}


void make_movement(command com) {
//  Serial.print("x: ");
//  Serial.println(com.x_steps);
//  Serial.print("y: ");
//  Serial.println(com.y_steps);
//  Serial.print("theta: ");
//  Serial.println(com.theta_steps);
//  Serial.print("z: ");
//  Serial.println(com.z_steps);

  if(com.x_steps!=0){
    digitalWrite(enablePinX,LOW);
  }
  if(com.y_steps!=0){
    digitalWrite(enablePinY,LOW);
  }
  if(com.theta_steps!=0){
    digitalWrite(enablePinTheta,LOW);
  }
  if(com.z_steps!=0){
    digitalWrite(enablePinZ,LOW);
  }
  if(com.pump_on==1){
    digitalWrite(pumpPin,LOW);
  } else{
    digitalWrite(pumpPin,HIGH);
  }
  if(com.l_on==1){
    digitalWrite(lightPin,LOW);
  } else{
    digitalWrite(lightPin,HIGH);
  }
  
  stepperX.setMaxSpeed(com.x_speed);
  stepperX.setAcceleration(com.x_acc);

  stepperY.setMaxSpeed(com.y_speed);
  stepperY.setAcceleration(com.y_acc);

  stepperTheta.setMaxSpeed(com.theta_speed);
  stepperTheta.setAcceleration(com.theta_acc);

  stepperZ.setMaxSpeed(com.z_speed);
  stepperZ.setAcceleration(com.z_acc);
  
  stepperX.move(com.x_steps);
  stepperY.move(com.y_steps);
  stepperZ.move(com.z_steps);
  stepperTheta.move(com.theta_steps);

  bool x_running = true;
  bool y_running = true;
  bool theta_running = true;
  bool z_running = true;


//  unsigned long curr_time = millis();

//  bool on = false;
   
  while (x_running || y_running|| theta_running || z_running){
    x_running = stepperX.run();
    y_running = stepperY.run();
    theta_running = stepperTheta.run();
    z_running = stepperZ.run();
  }

  // disable all devices
  digitalWrite(enablePinX, HIGH);
  digitalWrite(enablePinY, HIGH);
  digitalWrite(enablePinZ, HIGH);
  digitalWrite(enablePinTheta, HIGH);
}


void process_string() {
    command com;
    static byte ndx = 0;
    char endMarker = '\n';
    char rc;
   
    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (rc != endMarker) {
            receivedChars[ndx] = rc;
            ndx++;
            if (ndx >= numChars) {
                ndx = numChars - 1;
            }
        }
        else {
            receivedChars[ndx] = '\0'; // terminate the string
            ndx = 0;
            newData = true;
        }
    }

    if (newData) {
      com.x_steps = atoi(strtok(receivedChars, ":"));
      com.y_steps = atoi(strtok(NULL, ":"));
      com.theta_steps = atoi(strtok(NULL, ":"));
      com.z_steps = atoi(strtok(NULL, ":"));
      com.pump_on = atoi(strtok(NULL, ":"));
      com.l_on = atoi(strtok(NULL, ":"));
      com.x_speed = atoi(strtok(NULL, ":"));
      com.y_speed = atoi(strtok(NULL, ":"));
      com.theta_speed = atoi(strtok(NULL, ":"));
      com.z_speed = atoi(strtok(NULL, ":"));
      com.x_acc = atoi(strtok(NULL, ":"));
      com.y_acc = atoi(strtok(NULL, ":"));
      com.theta_acc = atoi(strtok(NULL, ":"));
      com.z_acc = atoi(strtok(NULL, ":"));
      
      newData=false;
      
      make_movement(com);

      Serial.println("End");
    }
}
