#include <Arduino_RouterBridge.h>

#define EEG_PIN_1 A0 
#define EEG_PIN_2 A1 

#define EMG_PIN_1 A2
#define EMG_PIN_2 A3

void setup() {
    Serial.begin(115200);
    Bridge.begin();

    Bridge.provide("readAnalogChannels", readAnalogChannels);

    // Register PinMode for Inputs
    pinMode(EEG_PIN_1, INPUT);
    pinMode(EEG_PIN_2, INPUT);
    pinMode(EMG_PIN_1, INPUT);
    pinMode(EMG_PIN_2, INPUT);

    analogReadResolution(12);
}

void loop() {
    Bridge.update();   // handle RPC from MPU
    delay(10);
}

String readAnalogChannels() {
    int a0 = analogRead(EEG_PIN_1);
    int a1 = analogRead(EEG_PIN_2);
    int a2 = analogRead(EMG_PIN_1);
    int a3 = analogRead(EMG_PIN_2);
    return String(a0) + "," + String(a1) + "," + String(a2) + "," + String(a3);
}

