"""
Simple script to capture mouse click positions
Captures 5 consecutive clicks and prints their coordinates
"""
from pynput import mouse
import sys

click_count = 0
max_clicks = 5

def on_click(x, y, button, pressed):
    global click_count
    if pressed and button == mouse.Button.left:
        click_count += 1
        print(f"Click {click_count}: x={x}, y={y}")
        
        if click_count >= max_clicks:
            print("\n=== All 5 clicks captured! ===")
            print("You can now close this window or press Ctrl+C")
            return False  # Stop listener

print("=== Mouse Click Position Capture ===")
print("Click anywhere on the screen 5 times...")
print("The coordinates will be printed below:\n")

# Start listening for clicks
with mouse.Listener(on_click=on_click) as listener:
    listener.join()

print("\nDone!")
