import json

if __name__ == '__main__':
    data = json.load(
        open(r'D:\PythonProject\IEMOCAP_Data\IEMOCAP_DATA\IEMOCAP_Audio_improve_Female_Session1.json', 'r'))
    print(data[0])
