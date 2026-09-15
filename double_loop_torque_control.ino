#include <Wire.h>
#include <SimpleFOC.h>


#define AS5600_ADDR 0x36
#define AS5600_CONF_HI 0x07

  /*
  T_SETTL1 Settling time SF = 00 2.2 ms
  T_SETTL2 Settling time SF = 01 1.1 ms
  T_SETTL3 Settling time SF = 10 0.55 ms
  T_SETTL4 Settling time SF = 11 0.286 ms
  */
void as5600_setFastFilter(TwoWire &bus) {
  bus.beginTransmission(AS5600_ADDR);
  bus.write(AS5600_CONF_HI);
  bus.endTransmission(false); // repeated start
  bus.requestFrom(AS5600_ADDR, 1);
  uint8_t confHi = bus.read();

  confHi &= 0b11111100;   // clear SF bits (1:0)
  confHi |= 0b00000011;   // SF = 11 (fastest, 0.286ms settle)

  bus.beginTransmission(AS5600_ADDR);
  bus.write(AS5600_CONF_HI);
  bus.write(confHi);
  bus.endTransmission();
}



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
  
  //set to fastest response minimal settling time
  as5600_setFastFilter(I2C_Encoder1);
  as5600_setFastFilter(I2C_Encoder2);

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
  motor.linkSensor(&sensor1);
  motor.linkDriver(&driver);

  motor.current_limit = 1.0;
  motor.voltage_limit = 6;
  motor.velocity_limit = 20;

  motor.controller = MotionControlType::torque; //velocity_openloop
  

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
float K = 1.3;//15
float I = 0.2;
float D = 3.3;
float first_joint_dependancy = 0.30;
float K2 = 0.6;//15
float I2 = 3;


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
float filter_alpha = 0.3;
float angle1 = 0;
float angle2 = 0;
float vel1_not_filtered = 0;
float vel2_not_filtered = 0;
int count = 0;
float pos_e_old1 = 0;
float pos_e_old2 = 0;
float pos_e_derivation1 = 0;
float pos_e_derivation2 = 0;
float vel_e1 = 0;
float vel_e2 = 0;
float vel_e_integration1 = 0;
float vel_e_integration2 = 0;
float torque_setpoint1 = 0;
float torque_setpoint2 = 0;





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
    count += 1;



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

    pos_e_derivation1 = (pos_e1 - pos_e_old1)*pid_HZ;
    pos_e_derivation2 = (pos_e2 - pos_e_old2)*pid_HZ;
    pos_e_old1 = pos_e1;
    pos_e_old2 = pos_e2;

    velocity_setpoint1 = K*(pos_e1 + I*pos_e_integration1 + D*pos_e_derivation1);
    velocity_setpoint2 = K*(pos_e2 + I*pos_e_integration2 + D*pos_e_derivation2);
    //velocity_setpoint = first_joint_dependancy*velocity_setpoint1 + velocity_setpoint2;

    vel_e1 = velocity_setpoint1 - vel1;
    vel_e2 = velocity_setpoint2 - vel2;

    vel_e_integration1 += vel_e1/pid_HZ;
    vel_e_integration2 += vel_e2/pid_HZ;


    torque_setpoint1 = K2*(vel_e1 + I2*vel_e_integration1 );
    torque_setpoint2 = K2*(vel_e2 + I2*vel_e_integration2 );

    motor.target = first_joint_dependancy*torque_setpoint1 + torque_setpoint2;
  
    
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
  if (millis() - lastPlot >= 1000/30) {
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
