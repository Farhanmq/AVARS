#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "model.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <TinyGPS++.h>
#include <SoftwareSerial.h>
#include <WiFi.h>
#include <ESP_Mail_Client.h>

// Choose the serial pins for ESP32 and GPS module
#define GPS_RX_PIN 16
#define GPS_TX_PIN 17

// SoftwareSerial pins for GSM module
#define GSM_TX_PIN 5
#define GSM_RX_PIN 4

// Pin for the push button
#define PUSH_BUTTON_PIN 19

// Create a TinyGPS++ object
TinyGPSPlus gps;

// Create a SoftwareSerial objects
SoftwareSerial gpsSerial(GPS_RX_PIN, GPS_TX_PIN);
SoftwareSerial gsmSerial(GSM_TX_PIN, GSM_RX_PIN);
const int buzzerPin = 18;             // Replace 12 with the actual GPIO pin number
const unsigned long interval = 1000;  // Delay between actions in milliseconds
unsigned long previousMillis = 0;     // Variable to store the previous time

Adafruit_MPU6050 mpu;
int samples = 0;
float ax, ay, az;
float baseAx, baseAy, baseAz;
float gx, gy, gz;
float baseGx, baseGy, baseGz;
const unsigned long measurementDuration = 572;  // 5 seconds
const unsigned long breakDuration = 5000;
tflite::ErrorReporter* tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 60 * 1024; // Reduced size
byte tensorArena[tensorArenaSize];

const char* GESTURES[] = {
  "accident",
  "non_accident"
};
#define NUM_GESTURES 2

// WiFi credentials
const char* ssid = "Mobile";
const char* password = "1234567i";

// Email credentials
#define SMTP_HOST "smtp.gmail.com"
#define SMTP_PORT 465
#define AUTHOR_EMAIL "codewhispers@gmail.com"
#define AUTHOR_PASSWORD "Poiuytr@1"
#define RECIPIENT_EMAIL "bazanahmad.BA@gmail.com"

// SMTP session and message objects
SMTPData smtpData;

void initTensorflow() {
  Serial.println("Initializing TensorFlow Lite...");
  // Get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  tflErrorReporter = &micro_error_reporter;

  static tflite::MicroInterpreter static_interpreter(
    tflModel, tflOpsResolver, tensorArena, tensorArenaSize, tflErrorReporter);

  tflInterpreter = &static_interpreter;

  // Allocate memory for the model's input and output tensors
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(tflErrorReporter, "AllocateTensors() failed");
    return;
  }

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.println("TensorFlow initialized");
}

void calibrateSensor() {
  const uint8_t numSamples = 100;  // Number of calibration samples

  float sumAccX = 0.0, sumAccY = 0.0, sumAccZ = 0.0;
  float sumGyroX = 0.0, sumGyroY = 0.0, sumGyroZ = 0.0;
  sensors_event_t a, g, temp;

  // Collect calibration samples
  for (int i = 0; i < numSamples; i++) {
    mpu.getEvent(&a, &g, &temp);
    sumAccX += a.acceleration.x;
    sumAccY += a.acceleration.y;
    sumAccZ += a.acceleration.z;
    sumGyroX += g.gyro.x;
    sumGyroY += g.gyro.y;
    sumGyroZ += g.gyro.z;
    delay(10);  // Add a small delay between readings to ensure stability
  }

  // Calculate calibration values as the average of the samples
  baseAx = sumAccX / numSamples;
  baseAy = sumAccY / numSamples;
  baseAz = sumAccZ / numSamples;
  baseGx = sumGyroX / numSamples;
  baseGy = sumGyroY / numSamples;
  baseGz = sumGyroZ / numSamples;
}

void setup() {
  Serial.begin(115200);  // Initialize the Serial monitor

  gpsSerial.begin(9600);  // Initialize the SoftwareSerial for GPS communication
  gsmSerial.begin(9600);  // Initialize the SoftwareSerial for GSM communication

  pinMode(PUSH_BUTTON_PIN, INPUT_PULLUP);  // Initialize the push button pin

  delay(1000);

  // Test communication with GSM module
  gsmSerial.println("AT");
  delay(1000);

  if (gsmSerial.find("OK")) {
    Serial.println("GSM module is ready");
  } else {
    Serial.println("GSM module is not responding");
  }

  // Set SMS text mode
  gsmSerial.println("AT+CMGF=1");
  delay(1000);

  pinMode(buzzerPin, OUTPUT);
  delay(1000);

  Serial.println("Adafruit MPU6050 test!");

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }

  Serial.println("MPU6050 Found!");
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  calibrateSensor();
  Serial.println("");

  Serial.print("Free heap before TensorFlow initialization: ");
  Serial.println(ESP.getFreeHeap());

  initTensorflow();

  Serial.print("Free heap after TensorFlow initialization: ");
  Serial.println(ESP.getFreeHeap());

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  delay(100);
}

void loop() {
  samples = 0;
  unsigned long elapsedTime = 0;

  while (elapsedTime < measurementDuration) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    ax = a.acceleration.x - baseAx;
    ay = a.acceleration.y - baseAy;
    az = a.acceleration.z - baseAz;
    gx = g.gyro.x - baseGx;
    gy = g.gyro.y - baseGy;
    gz = g.gyro.z - baseGz;

    tflInputTensor->data.f[samples * 6 + 0] = (ax + 4.0) / 8.0;
    tflInputTensor->data.f[samples * 6 + 1] = (ay + 4.0) / 8.0;
    tflInputTensor->data.f[samples * 6 + 2] = (az + 4.0) / 8.0;
    tflInputTensor->data.f[samples * 6 + 3] = (gx + 2000) / 4000;
    tflInputTensor->data.f[samples * 6 + 4] = (gy + 2000) / 4000;
    tflInputTensor->data.f[samples * 6 + 5] = (gz + 2000) / 4000;
    elapsedTime++;
    samples++;
  }

  Serial.println(samples);
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    return;
  }

  Serial.println(tflOutputTensor->data.f[0] > 0.7, 6);
  if (tflOutputTensor->data.f[0] > 0.5) {
    unsigned long startTime = millis();
    while (millis() - startTime < 10000) {
      if (digitalRead(PUSH_BUTTON_PIN) == LOW) {  // Check if the button is pressed
        Serial.println("Button pressed, resetting loop");
        return;  // Exit the loop to reset
      }

      unsigned long currentMillis = millis();
      if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis;

        digitalWrite(buzzerPin, HIGH);  // Turn on the buzzer
        delay(100);                     // Buzz duration (adjust as needed)
        digitalWrite(buzzerPin, LOW);   // Turn off the buzzer
      }
    }
    while (gpsSerial.available() > 0) {
      char data = gpsSerial.read();
      gps.encode(data);
    }

    // Check if new data is available and parsed successfully
    float latitude = 31.4435857;
    float longitude = 74.3164413;

    String mapsLink = "https://www.google.com/maps/search/?api=1&query=" + String(latitude, 6) + "," + String(longitude, 6);
    Serial.println("Google Maps link: " + mapsLink);

    // Send GSM SMS
    gsmSerial.println("AT+CMGS=\"+923164532636\"");  // Replace with the desired phone number
    delay(1000);

    gsmSerial.println("Accident Detected!!!");
    gsmSerial.print(mapsLink);

    gsmSerial.write(26);

    // Send email notification
    smtpData.setLogin(SMTP_HOST, SMTP_PORT, AUTHOR_EMAIL, AUTHOR_PASSWORD);
    smtpData.setSender("ESP32", AUTHOR_EMAIL);
    smtpData.setPriority("High");
    smtpData.setSubject("Accident Detected!!!");
    smtpData.setMessage("Accident Detected!!!\nGoogle Maps link: " + mapsLink, false);
    smtpData.addRecipient("Recipient", RECIPIENT_EMAIL);

    // Connect and send the email
    if (!MailClient.sendMail(smtpData)) {
      Serial.println("Error sending email, " + MailClient.smtpErrorReason());
    } else {
      Serial.println("Email sent successfully!");
    }

    // Clear all data from the SMTP session to free up memory
    smtpData.empty();
  }
}
