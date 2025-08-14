#define trigPin 9
#define echoPin 10
#define relayPin 13
#define buzzerPin 12

const int ON_DISTANCE = 20;   // Turn ON if object is closer than this
const int OFF_DISTANCE = 25;  // Turn OFF if object is farther than this

bool bulbOn = false;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(relayPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);

  digitalWrite(relayPin, LOW);  // Relay OFF at start
  digitalWrite(buzzerPin, LOW); // Buzzer OFF at start

  Serial.begin(9600);
}

long getDistanceCM() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 30000); // Timeout at ~5m
  if (duration == 0) return 9999; // No reading
  return duration * 0.034 / 2;
}

void loop() {
  long distance = getDistanceCM();

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  if (!bulbOn && distance < ON_DISTANCE && distance > 0) {
    bulbOn = true;
  }
  else if (bulbOn && distance > OFF_DISTANCE) {
    bulbOn = false;
  }

  if (bulbOn) {
    digitalWrite(relayPin, HIGH);  // Bulb ON
    digitalWrite(buzzerPin, HIGH); // Buzzer ON
  } else {
    digitalWrite(relayPin, LOW);   // Bulb OFF
    digitalWrite(buzzerPin, LOW);  // Buzzer OFF
  }

  delay(100);
}
