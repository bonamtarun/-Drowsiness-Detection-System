bool blinkLED = false;
bool ledState = false;
char sw;

void setup() 
{
  Serial.begin(9600);
  pinMode(13, OUTPUT);   // Blinking LED
  pinMode(6, OUTPUT);    // Motor control
  pinMode(11, OUTPUT);    // Second LED when switch is ON
  pinMode(9, INPUT_PULLUP);     // Switch input

  digitalWrite(6, LOW);  // Motor ON initially
  digitalWrite(13, LOW); // LED OFF
  digitalWrite(11, LOW);  // Second LED OFF
}

void loop() {
  sw = digitalRead(9);  // Read switch state

  if (sw == HIGH)
   {
    digitalWrite(11, HIGH);   // Second LED ON
    if (!blinkLED) 
    {
      digitalWrite(6, LOW);  // Motor ON (unless blinking)
    }

    if (Serial.available()) 
    {
      char state = Serial.read();
      if (state == '1') 
      {
        blinkLED = true;     // Start blinking
      } else if (state == '0') 
      {
        blinkLED = false;    // Stop blinking
        digitalWrite(13, LOW);
        ledState = false;
        digitalWrite(6, LOW);  // Motor ON
      }
    }

    if (blinkLED) 
    {
      digitalWrite(6, HIGH);     // Turn motor OFF while blinking
      ledState = !ledState;
      digitalWrite(13, ledState); // Blink LED
      delay(50);
    }

  }
   else
    {
    // Switch OFF: reset all
    blinkLED = false;
    ledState = false;
    digitalWrite(11, LOW);    // Second LED OFF
    digitalWrite(13, LOW);   // Blinking LED OFF
    digitalWrite(6, HIGH);   // Motor OFF
  }
}
