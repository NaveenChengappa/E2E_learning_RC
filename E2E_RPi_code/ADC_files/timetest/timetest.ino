double channel_1,channel_2;
void setup() {
  // Pin 7 : Steering read pin
  // Pin 2 : Throttle read pin
  // Baud rate : 9600
  pinMode(7, INPUT);
  pinMode(2, INPUT);
  Serial.begin(9600);
}

void loop() {
  // Read pin 7 and 2   
  channel_2 = pulseIn(7, HIGH);
  channel_1 = pulseIn(2, HIGH);
  delay(70); 
  // Output onto serial monitor       
  Serial.print(channel_2);
  Serial.println(channel_1); 
}
