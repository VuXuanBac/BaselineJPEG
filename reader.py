def get_type(t: int) -> str:
    # if t != 0xFF:
    #     return 'ECS'                  # Entropy-coded Segment
    if t == 0xDA: return 'SOS'          # Start of Scan
    if t == 0xDB: return 'DQT'          # Definition of Quantization Table
    if t == 0xD8: return 'SOI'          # Start of Image
    if t == 0xD9: return 'EOI'          # End of Image
    if 0xD0 <= t <= 0xD7:
        return 'RST'                    # Restart Interval
    if 0xC0 <= t <= 0xCF:
        if t == 0xC4: return 'DHT'      # Definition of Huffman Table
        if t == 0xC8: return 'JPG'      # JPEG Extension
        if t == 0xCC: return 'DAC'      # Definition of Arithmetic
        return 'SOF'                    # Start of Frame
    if 0xE0 <= t <= 0xEF:               
        return 'APP'                    # Application
    if 0xF0 <= t <= 0xFD:
        return 'JPG'                    # JPEG Extension
    if t == 0xDC: return 'DNL'          # Define New Line
    if t == 0xDD: return 'DRI'          # Define Restart Interval
    if t == 0xDE: return 'DHP'          # 
    if t == 0xDF: return 'EXP'          #     
    if t == 0xFE: return 'COM'          # Comment   
    # if t <= 0xBF:
    return '***'                        # Reserved

def read_one_byte(fp):
    b = fp.read(1)
    if b == None or len(b) < 1:
        raise Exception('EOF')
    return b[0]

def read_ecs(fp) -> tuple[bytes, int]:
    '''
    Read byte by byte in ECS, remove any stuff (0x00 followed 0xFF)
    '''
    data = bytearray()
    stuff = 0
    
    while True:
        b = read_one_byte(fp)
        if b == 0xFF:
            b2 = read_one_byte(fp)
            if b2 != 0:
                break # Any markers 0xFFxx
            else:
                stuff += 1
        data.append(b)
    fp.seek(-2, 1)
    return bytes(data), stuff

def get_ecs_length(path) -> int:
    result = 0
    with open(path, 'rb') as file:
        try:
            while True:
                b0 = read_one_byte(file)
                if b0 == 0xFF:
                    b1 = read_one_byte(file)
                    if b1 != 0:
                        type = get_type(b1)
                        if type == 'EOI':
                            break
                        if type not in ['SOI', 'EOI', 'RST', 'ECS']:
                            length = int.from_bytes(file.read(2), 'big')
                            start = file.tell()
                            file.seek(length - 2, 1)
                            # print(f'{type}: {length} bytes start {start - 2:0x} end {file.tell():0x}')
                        else:
                            # print(f'{type}')
                            pass
                        continue
                    else: # FF00
                        file.seek(-2, 1) # precede 2 bytes
                else:
                    file.seek(-1, 1) # precede 1 byte
                ecs, stuff = read_ecs(file)
                result += len(ecs)
                # print(f'Segment ECS has length {len(ecs)}, stuff {stuff}')
        except:
            pass
    return result
