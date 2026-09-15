#include <Wire.h>
#include <SimpleFOC.h>

// ============================================================
// MOTOR
// ============================================================

BLDCMotor motor = BLDCMotor(7, 17);
BLDCDriver3PWM driver = BLDCDriver3PWM(25, 26, 27, 14);

// ============================================================
// I2C BUSES (ESP32 has 2 hardware I2C controllers)
// ============================================================

TwoWire I2C_Encoder1 = TwoWire(0);
TwoWire I2C_Encoder2 = TwoWire(1);

// ============================================================
// TWO AS5600 SENSORS
// ============================================================

MagneticSensorI2C sensor1 = MagneticSensorI2C(AS5600_I2C);
MagneticSensorI2C sensor2 = MagneticSensorI2C(AS5600_I2C);

float angle1_zero = 0;
float angle2_zero = 0;
// ============================================================
// COMMANDER
// ============================================================

Commander command = Commander(Serial);

void doTarget(char* cmd) {
  command.scalar(&motor.target, cmd);
}

void doLimit(char* cmd) {
  command.scalar(&motor.voltage_limit, cmd);
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("Starting system...");

  // -----------------------------
  // I2C setup
  // -----------------------------
  I2C_Encoder2.begin(21, 22);  // encoder 2
  I2C_Encoder1.begin(16, 17);  // encoder 1

  // -----------------------------
  // Sensors init
  // -----------------------------
  sensor1.init(&I2C_Encoder1);
  sensor2.init(&I2C_Encoder2);

  // allow sensor to read real value
  delay(50);

  sensor1.update();
  sensor2.update();

  angle1_zero = sensor1.getAngle();
  angle2_zero = sensor2.getAngle();

  Serial.println("Encoders ready");

  // -----------------------------
  // Motor driver
  // -----------------------------
  driver.voltage_power_supply = 13.8;
  driver.voltage_limit = 8;

  if (!driver.init()) {
    Serial.println("Driver FAILED");
    return;
  }

  // -----------------------------
  // Motor setup
  // -----------------------------
  //motor.linkSensor(&sensor1);
  motor.linkDriver(&driver);

  motor.current_limit = 1.0;
  motor.voltage_limit = 6;
  motor.velocity_limit = 20;

  motor.controller = MotionControlType::velocity_openloop;
  

  if (!motor.init()) {
    Serial.println("Motor FAILED");
    return;
  }
  if (!motor.initFOC()) {
    Serial.println("Motor FAILED");
   return;
  }
  

  motor.target = 0;

  // -----------------------------
  // Commands
  // -----------------------------
  command.add('T', doTarget, "target velocity");
  command.add('L', doLimit,  "voltage limit");

  Serial.println("System ready");
}

// ============================================================
// LOOP
// ============================================================
int stop = 0;
float pid_HZ = 500.0;
float setpoint_1 = 0.0;
float setpoint_2 = 0.0;
float K = 2;//15
float I = 0.5;
float D = 2;
float first_joint_dependancy = 0.30;

float pos_e1 = 0;
float pos_e2 = 0;
float pos_e_integration1 = 0;
float pos_e_integration2 = 0;
float velocity_setpoint1 = 0;
float velocity_setpoint2 = 0;
float velocity_setpoint = 0;
float angle1_filtered = 0;
float angle2_filtered = 0;
float angle1_filtered_old = 0;
float angle2_filtered_old = 0;
float vel1 = 0;
float vel2 = 0;
float filter_alpha = 0.6;
float angle1 = 0;
float angle2 = 0;
float vel1_not_filtered = 0;
float vel2_not_filtered = 0;


unsigned long diff = 0;
unsigned long lastPlot = 0;
unsigned long lastPID = micros();
unsigned long start = micros();
void loop() {
  // -----------------------------
  // update encoders
  // -----------------------------
  
  
  

  //pid controll
  
  if(micros() - lastPID >= 1000000/pid_HZ){
    diff = micros() - lastPID;
    lastPID += 1000000/pid_HZ;




    sensor1.update();
    sensor2.update();

    angle1 = sensor1.getAngle() - angle1_zero;
    //vel1_not_filtered  = sensor1.getVelocity();


    angle2 = sensor2.getAngle() - angle2_zero;
    //vel2_not_filtered  = sensor2.getVelocity();


    
    angle1_filtered += filter_alpha*(angle1-angle1_filtered);
    angle2_filtered += filter_alpha*(angle2-angle2_filtered);

    vel1 = (angle1_filtered - angle1_filtered_old)*pid_HZ;
    vel2 = (angle2_filtered - angle2_filtered_old)*pid_HZ;
    angle1_filtered_old = angle1_filtered;
    angle2_filtered_old = angle2_filtered;




    pos_e1 = setpoint_1 - angle1_filtered;
    pos_e2 = setpoint_2 - angle2_filtered;

    
    pos_e_integration1 += pos_e1/pid_HZ;
    pos_e_integration2 += pos_e2/pid_HZ;

    velocity_setpoint1 = K*(pos_e1 + I*pos_e_integration1 - D*vel1);
    velocity_setpoint2 = K*(pos_e2 + I*pos_e_integration2 - D*vel2);
    velocity_setpoint = first_joint_dependancy*velocity_setpoint1 + velocity_setpoint2;
    motor.target = -velocity_setpoint;


    if (angle1 > 1.5 || angle1 < -1.5 || angle2 > 1 || angle2 < -1 || stop == 1){
      motor.target = 0;
      stop = 1;
    }
  }
  





  // -----------------------------
  // motor loop
  // -----------------------------
  
  motor.loopFOC();
  motor.move();

  command.run();

  // -----------------------------
  // plotting
  // -----------------------------
  if (millis() - lastPlot >= 1000/10) {
    lastPlot = millis();

    Serial.print("micros:");
    Serial.println(diff);
    Serial.print(",");

    Serial.print("A1:");
    Serial.print(angle1, 5);
    Serial.print(",");

    Serial.print("V1:");
    Serial.print(vel1, 5);
    Serial.print(",");

    Serial.print("A2:");
    Serial.print(angle2, 5);
    Serial.print(",");

    Serial.print("V2:");
    Serial.print(vel2, 5);
    Serial.print(",");

    Serial.print("Target:");
    Serial.println(motor.target, 5);
    

    
  }
  
}
