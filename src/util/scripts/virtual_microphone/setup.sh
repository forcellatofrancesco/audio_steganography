# Load a null sink
pactl load-module module-null-sink sink_name=virtual_mic sink_properties=device.description="Virtual_Mic"

# Create a virtual source FROM the monitor — this makes it look like a real mic
pactl load-module module-virtual-source source_name=virtual_mic_source master=virtual_mic.monitor source_properties=device.description="Virtual_Mic_Source"