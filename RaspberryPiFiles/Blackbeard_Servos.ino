#include <Servo.h>
#include <string.h>

Servo frontservo;
Servo backservo;

int front_angle;
int back_angle;

int FRONT_OPEN_ANGLE = 105;
int FRONT_CLOSE_ANGLE = 45;
int BACK_OPEN_ANGLE = 165;
int BACK_CLOSE_ANGLE = 225;

float US_TO_PULSE = 2000/270;

int angle_to_pwm(int input_angle){
  // converts 270 angle to 180, so that the same PWM can be applied
  return (input_angle * US_TO_PULSE) + 500;
}

void move_servo(int new_angle, char frontOrBack){

    int pulse_width = angle_to_pwm(new_angle);

    // init local variables
    Servo servoMotor;
    int old_angle;

    // check if the front or back servo will be moved
    if(frontOrBack=='f'){
      servoMotor = frontservo;
      old_angle = front_angle;
    } else if(frontOrBack=='b'){
      servoMotor = backservo;
      old_angle = back_angle;
    }

    while(new_angle != old_angle) {

      if(new_angle > old_angle) {
        // INCREASING SERVO ANGLE
        old_angle += 1;
      } else if(new_angle < old_angle) {
        // DECREASING SERVO ANGLE
        old_angle -= 1;
      }
      // move servo and sleep for set time
      servoMotor.writeMicroseconds(angle_to_pwm(old_angle));
      delay(50); 
    }

    // UPDATE ANGLE VARIABLE
    if(frontOrBack=='f'){
      front_angle = old_angle;
    } else if(frontOrBack=='b'){
      back_angle = old_angle;
    }
    
}

void setup() {
  Serial.begin(115200);

  frontservo.attach(14);  // gpio17
  backservo.attach(17);  // gpio15
  

  // init servos
  frontservo.write(angle_to_pwm(FRONT_CLOSE_ANGLE));
  backservo.write(angle_to_pwm(BACK_CLOSE_ANGLE));

  front_angle = FRONT_CLOSE_ANGLE;
  back_angle = BACK_CLOSE_ANGLE;

}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available() > 0) {
    String serialData = Serial.readString();
    //serialData = Serial.readStringUntil(END_SERIAL_READ);
    Serial.println(serialData);
    for(int i = 0; i < serialData.length(); i++) {
      Serial.write(serialData[i]);
    }

    if(serialData=="front_open"){
      move_servo(FRONT_OPEN_ANGLE, 'f');
    } else if(serialData=="front_close"){
      move_servo(FRONT_CLOSE_ANGLE, 'f');
    } else if(serialData=="back_open"){
      move_servo(BACK_OPEN_ANGLE, 'b');
    } else if(serialData=="back_close"){
      move_servo(BACK_CLOSE_ANGLE, 'b');
    }

  }
}
