def adjust_gcode_speeds(input_gcode, output_gcode, repositioning_speed=5000, plotting_speed=500):
    """
    Adjust the speeds for repositioning (G0) and plotting (G1) movements in the G-code.
    :param input_gcode: Path to input G-code file.
    :param output_gcode: Path to save the modified G-code file.
    :param repositioning_speed: Feed rate for G0 (repositioning) moves.
    :param plotting_speed: Feed rate for G1 (plotting) moves.
    """
    with open(input_gcode, 'r') as infile, open(output_gcode, 'w') as outfile:
        for line in infile:
            if line.startswith("G0"):  # Repositioning move
                outfile.write(f"{line.strip()} F{repositioning_speed}\n")
            elif line.startswith("G1"):  # Plotting move
                outfile.write(f"{line.strip()} F{plotting_speed}\n")
            else:
                outfile.write(line)

    print(f"Modified G-code saved to {output_gcode}")
