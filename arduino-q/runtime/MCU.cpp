#include <Arduino_RouterBridge.h>

#define MODE_MAIN 0
#define MODE_TRAINING 1
#define MODE MODE_MAIN

// Training: sampling is driven by MCP at COLLECT_INTERVAL_SEC (e.g. 100 Hz)
#define TRAINING_SAMPLE_INTERVAL_MS 10

// Runtime: keep last n samples for ML window (getRecentWindow)
#define RUNTIME_RING_BUFFER_SIZE 300

// EEG mode: 0 = amp (moving avg), 1 = ML (model inference on window)
#define EEG_MODE_AMP 0
#define EEG_MODE_ML 1
#define EEG_MODE EEG_MODE_AMP

// Motor 1 (steering): SimpleFOC shield — 3 PWM (phases A,B,C) + Enable
#define MOTOR1_PWM_A 9
#define MOTOR1_PWM_B 5
#define MOTOR1_PWM_C 6
#define MOTOR1_ENABLE 8

// Speed wheel: single PWM (e.g. DC motor or second BLDC)
#define SPEED_WHEEL_PWM 3

#define EEG_PIN_1 A0
#define EEG_PIN_2 A1
#define EMG_PIN_1 A2
#define EMG_PIN_2 A3

// Ring buffer: [RUNTIME_RING_BUFFER_SIZE][4] (samples x channels A0,A1,A2,A3)
static uint16_t ringBuf[RUNTIME_RING_BUFFER_SIZE][4];
static int ringWriteIndex = 0;
static int ringCount = 0;

String readAnalogChannels();
String getRecentWindow();
String getConfig();
void setSteering(String normalizedAmplitudeStr);
void setSpeedWheel(String normalizedSpeedStr);

void setup() {
    Serial.begin(115200);

    if (!Bridge.begin()) {
        while (true) {}
    }

    pinMode(EEG_PIN_1, INPUT);
    pinMode(EEG_PIN_2, INPUT);
    pinMode(EMG_PIN_1, INPUT);
    pinMode(EMG_PIN_2, INPUT);
    analogReadResolution(12);

    analogWriteResolution(12);

    pinMode(MOTOR1_ENABLE, OUTPUT);
    digitalWrite(MOTOR1_ENABLE, LOW);
    pinMode(MOTOR1_PWM_A, OUTPUT);
    pinMode(MOTOR1_PWM_B, OUTPUT);
    pinMode(MOTOR1_PWM_C, OUTPUT);
    pinMode(SPEED_WHEEL_PWM, OUTPUT);

    if (!Bridge.provide("readAnalogChannels", readAnalogChannels)) {
        while (true) {}
    }
    if (!Bridge.provide("getRecentWindow", getRecentWindow)) {
        while (true) {}
    }
    if (!Bridge.provide("getConfig", getConfig)) {
        while (true) {}
    }
    if (!Bridge.provide("setSteering", setSteering)) {
        while (true) {}
    }
    if (!Bridge.provide("setSpeedWheel", setSpeedWheel)) {
        while (true) {}
    }
}

void loop() {
    delay(10);
}

String readAnalogChannels() {
    int a0 = analogRead(EEG_PIN_1);
    int a1 = analogRead(EEG_PIN_2);
    int a2 = analogRead(EMG_PIN_1);
    int a3 = analogRead(EMG_PIN_2);

    ringBuf[ringWriteIndex][0] = (uint16_t)a0;
    ringBuf[ringWriteIndex][1] = (uint16_t)a1;
    ringBuf[ringWriteIndex][2] = (uint16_t)a2;
    ringBuf[ringWriteIndex][3] = (uint16_t)a3;
    ringWriteIndex = (ringWriteIndex + 1) % RUNTIME_RING_BUFFER_SIZE;
    if (ringCount < RUNTIME_RING_BUFFER_SIZE) {
        ringCount++;
    }

    String out = String(a0) + "," + String(a1) + "," + String(a2) + "," + String(a3);
#if MODE == MODE_TRAINING
    Serial.println(out);
#endif
    return out;
}

String getRecentWindow() {
    // Return oldest-to-newest: start at (ringWriteIndex - ringCount + N) % N
    String result = "";
    int n = ringCount;
    if (n == 0) return result;
    int start = (ringWriteIndex - n + RUNTIME_RING_BUFFER_SIZE) % RUNTIME_RING_BUFFER_SIZE;
    for (int i = 0; i < n; i++) {
        int idx = (start + i) % RUNTIME_RING_BUFFER_SIZE;
        if (i > 0) result += ",";
        result += String(ringBuf[idx][0]) + "," + String(ringBuf[idx][1]) + "," +
                  String(ringBuf[idx][2]) + "," + String(ringBuf[idx][3]);
    }
    return result;
}

String getConfig() {
    // "eeg_mode,ring_buffer_size"
    return String(EEG_MODE) + "," + String(RUNTIME_RING_BUFFER_SIZE);
}

// SimpleFOC: 3-phase PWM (9,5,6) + Enable (8). Open-loop: magnitude and direction.
// For full SimpleFOC use BLDCMotor + driver library when available.
void setSteering(String normalizedAmplitudeStr) {
    float val = normalizedAmplitudeStr.toFloat();
    float mag = (val < 0) ? -val : val;
    if (mag > 1.0f) mag = 1.0f;
    int pwmVal = (int)(mag * 4095.0f);
    if (pwmVal > 4095) pwmVal = 4095;
    digitalWrite(MOTOR1_ENABLE, (pwmVal > 0) ? HIGH : LOW);
    if (val >= 0) {
        analogWrite(MOTOR1_PWM_A, pwmVal);
        analogWrite(MOTOR1_PWM_B, 0);
        analogWrite(MOTOR1_PWM_C, 0);
    } else {
        analogWrite(MOTOR1_PWM_A, 0);
        analogWrite(MOTOR1_PWM_B, pwmVal);
        analogWrite(MOTOR1_PWM_C, 0);
    }
}

void setSpeedWheel(String normalizedSpeedStr) {
    float val = normalizedSpeedStr.toFloat();
    if (val < 0) val = 0;
    if (val > 1.0f) val = 1.0f;
    int pwmVal = (int)(val * 4095.0f);
    if (pwmVal > 4095) pwmVal = 4095;
    analogWrite(SPEED_WHEEL_PWM, pwmVal);
}
