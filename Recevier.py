import RPi.GPIO as GPIO
import time
import serial


ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=0.1)


servo_pins = {
    1: 17,  
    2: 18,  
    3: 27,  
    4: 22   
}

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


servos = {}
for dot, pin in servo_pins.items():
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, 50)  
    pwm.start(0)
    servos[dot] = pwm


led_pin = 23
buzzer_pin = 24
GPIO.setup(led_pin, GPIO.OUT)
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.output(led_pin, GPIO.LOW)
GPIO.output(buzzer_pin, GPIO.LOW)



def move_servo(dot, active):
    angle = 90 if active else 0
    duty = 2.5 + (angle / 180.0) * 10
    servos[dot].ChangeDutyCycle(duty)
    time.sleep(0.3)
    servos[dot].ChangeDutyCycle(0)  

def reset_all_servos():
    for dot in servos:
        move_servo(dot, False)

braille_map = {
    "1": [1],
    "2": [1, 2],
    "3": [1, 3],
    "4": [1, 4,3],
    "5": [1, 4],
    "6": [1, 2, 3],
    "7": [1, 2, 3, 4],
    "8": [1, 2, 3],
    "9": [2, 4],
    "10": [2, 3, 4]
}

print("Bluetooth Braille Servo Controller Ready...")

try:
    while True:
        if ser.in_waiting > 0:
            try:
                data = ser.read(ser.in_waiting).decode(errors='ignore').strip()
            except UnicodeDecodeError:
                continue

            if not data:
                continue

            print(f"Received: {data}")

            GPIO.output(led_pin, GPIO.HIGH)
            GPIO.output(buzzer_pin, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(led_pin, GPIO.LOW)
            GPIO.output(buzzer_pin, GPIO.LOW)

            
            reset_all_servos()

            
            if data in braille_map:
                active_dots = braille_map[data]
                for dot in active_dots:
                    move_servo(dot, True)
            else:
                print("Invalid Braille number received")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Shutting down...")

finally:
    for pwm in servos.values():
        pwm.stop()
    GPIO.cleanup()
