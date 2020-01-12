#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define pumpPin 0
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

const byte numChars = 32;
char receivedChars[numChars];   // an array to store the received data

boolean newData = false;

void setup() {
    Serial.begin(9600);
    Serial.println("<Arduino is ready>");

    for(int i = 0; i < 14; i++) {
      pinMode(i, OUTPUT);
    }

    // disable all other devices
    digitalWrite(enablePinX, HIGH);
    digitalWrite(enablePinY, HIGH);
    digitalWrite(enablePinZ, HIGH);
    digitalWrite(enablePinTheta, HIGH);
    digitalWrite(pumpPin, HIGH);
}

void loop() {
    recvWithEndMarker();
    showNewData();
}

void recvWithEndMarker() {
    static byte ndx = 0;
    char endMarker = '\n';
    char rc;
   
    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();
        Serial.print(rc);

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
}


void move_stepper(char motor, int dir, int steps, int delay_time){
  // disable all other devices
  digitalWrite(enablePinX, HIGH);
  digitalWrite(enablePinY, HIGH);
  digitalWrite(enablePinZ, HIGH);
  digitalWrite(enablePinTheta, HIGH);

  int step_pin = stepPinX;
  int dir_pin = dirPinX;

  if(motor == 'X') {
    digitalWrite(enablePinX, LOW);   
    dir_pin = dirPinX;
    step_pin = stepPinX;
  
  } else if(motor == 'Y') {
    digitalWrite(enablePinY, LOW);   
    dir_pin = dirPinY;
    step_pin = stepPinY;

  } else if(motor == 'Z') {
    digitalWrite(enablePinZ, LOW); 
    dir_pin = dirPinZ;
    step_pin = stepPinZ;
    
  } else if(motor == 'R') {
    digitalWrite(enablePinTheta, LOW); 
    dir_pin = dirPinTheta;
    step_pin = stepPinTheta;
  }

  // Set Direction
  if (dir == 1) {
    digitalWrite(dir_pin, HIGH); 
  } else {
    digitalWrite(dir_pin, LOW); 
  }

  // Spin the stepper motor 1 revolution 
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(step_pin, HIGH);
    delayMicroseconds(delay_time);
    digitalWrite(step_pin, LOW);
    delayMicroseconds(delay_time);
  }

  digitalWrite(enablePinX, HIGH);
  digitalWrite(enablePinY, HIGH);
  digitalWrite(enablePinZ, HIGH);
  digitalWrite(enablePinTheta, HIGH);
}


void showNewData() {
    if (newData == true) {
        Serial.print("Arduino Output: ");
        Serial.println(receivedChars);
        
        char *pt;
        pt = strtok(receivedChars, ":");

        char motor = atoi(pt);
        int mode = atoi(strtok(NULL, ":"));

        if (motor == 'P'){
          digitalWrite(pumpPin, mode);
          Serial.print("PUMPING");
          Serial.println(mode);
        } else {
          move_stepper(motor, (mode > 0), abs(mode), 1000);
        }
        
        newData = false;
    }
}
