# streamlit run web_deploy.py
# pip freeze > requirements.txt
# pip install -r requirements.txt

import pickle

#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from lightgbm import LGBMRegressor


#load the model from disk
#import joblib
#filename = 'finalized_model.pkl'
#model = joblib.load(filename)

#load the model from disk
import pickle
filename = 'finalized_model.pkl'
model= pickle.load(open(filename,"rb"))
#model= pickle.load(open("/content/drive/MyDrive/Colab Notebooks/saved_model.pkl","rb"))

def main():
    #Setting Application title
    st.title('Fenyx 4DATA AutoScout Model App')
    st.write('Hello, *ALL!* :sunglasses:')

    #Setting Application description
    st.markdown("""
    :dart:  This Streamlit app is made to predict the AutoScout use cases.
    The application is functional for both online prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('App.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict the AutoScout use cases')
    st.sidebar.image(image)



    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection

        my_dict=dict()

        st.subheader("Make & Model") 
        make_model_dic= {"Audi-A1":0,"Audi-A3":1,"Audi-A4":2,"Audi-A4allroad":3,"Audi-A5":4,"Audi-A6":5,"Audi-A6allroad":6,"Audi-A7":7,"Audi-A8":8,"Audi-Others":9,"Audi-Q2":10,	"Audi-Q3":11,	"Audi-Q4e-tron":12,	"Audi-Q5":13,	"Audi-Q7":14,	"Audi-Q8":15,	"Audi-R8":16,	"Audi-S5":17,	"Audi-SQ5":18,	"Audi-SQ7":19,	"Audi-TT":20,	"Audi-e-tron":21,	"Audi-e-tronGT":22,	"BMW-116":23,	"BMW-118":24,	"BMW-120":25,	"BMW-218":26,	"BMW-220":27,	"BMW-225":28,	"BMW-316":29,	"BMW-318":30,	"BMW-320":31,	"BMW-325":32,	"BMW-328":33,	"BMW-330":34,	"BMW-335":35,	"BMW-420":36,	"BMW-428":37,	"BMW-435":38,	"BMW-520":39,	"BMW-523":40,	"BMW-525":41,	"BMW-530":42,	"BMW-535":43,	"BMW-640":44,	"BMW-650":45,	"BMW-730":46,	"BMW-740":47,	"BMW-M3":48,	"BMW-M4":49,	"BMW-Others":50,	"BMW-X1":51,	"BMW-X2":52,	"BMW-X3":53,	"BMW-X5":54,	"BMW-X6":55,	"BMW-Z3":56,	"BMW-Z4":57,	"BMW-i3":58,	"BMW-i4":59,	"BMW-iX":60,	"BMW-iX3":61,	"Chevrolet-Aveo":62,	"Chevrolet-Camaro":63,	"Chevrolet-Captiva":64,	"Chevrolet-Corvette":65,	"Chevrolet-Cruze":66,	"Citroen-Berlingo":68,	"Citroen-C2":70,	"Citroen-C3Picasso":72,	"Citroen-C4":73,	"Citroen-C4Picasso":74,	"Citroen-C5":75,	"Citroen-DS3":77,	"Citroen-GrandC4Picasso":78,	"Citroen-Jumpy":79,	"Citroen-Others":80,	"Citroen-XsaraPicasso":81,	"Dacia-Duster":82,	"Dacia-Logan":83,	"Fiat-500":85,	"Fiat-500C":86,	"Fiat-500L":87,	"Fiat-Doblo":90,	"Fiat-PuntoEvo":95,	"Fiat-Stilo":96,	"Ford-C-Max":97,	"Ford-Explorer":98,	"Ford-F150":99,	"Ford-Fiesta":100,	"Ford-Focus":101,	"Ford-FocusC-Max":102,	"Ford-FocusCC":103,	"Ford-Fusion":104,	"Ford-Galaxy":105,	"Ford-Ka/Ka+":106,	"Ford-Kuga":107,	"Ford-Mondeo":108,	"Ford-Mustang":109,	"Ford-MustangMach-E":110,	"Ford-Puma":111,	"Ford-Ranger":112,	"Ford-S-Max":113,	"Ford-Transit":114,	"Ford-TransitConnect":115,	"Ford-TransitCustom":116,	"Honda-Accord":117,	"Honda-CR-V":118,	"Honda-Civic":119,	"Honda-Jazz":121,	"Hyundai-Getz":123,	"Hyundai-Ioniq5":125,	"Hyundai-Kona":126,	"Hyundai-Nexo":127,	"Hyundai-i10":130,	"Hyundai-i20":131,	"Hyundai-i30":132,	"Hyundai-iX35":134,	"Kia-CeedSW/ceedSW":136,	"Kia-Ceed":137,	"Kia-EV6":138,	"Kia-Optima":140,	"Kia-Picanto":141,	"Kia-Sorento":144,	"Kia-Soul":145,	"Kia-Sportage":146,	"Kia-Venga":147,	"Kia-XCeed":148,	"Mazda-2":150,	"Mazda-5":152,	"Mazda-6":153,	"Mazda-CX-3":154,	"Mazda-CX-30":155,	"Mercedes-Benz-A150":159,	"Mercedes-Benz-A160":160,	"Mercedes-Benz-A170":161,	"Mercedes-Benz-A200":162,	"Mercedes-Benz-A250":163,	"Mercedes-Benz-A45AMG":164,	"Mercedes-Benz-B170":165,	"Mercedes-Benz-B180":166,	"Mercedes-Benz-B200":167,	"Mercedes-Benz-B250":168,	"Mercedes-Benz-C180":169,	"Mercedes-Benz-C200":170,	"Mercedes-Benz-C220":171,	"Mercedes-Benz-C250":172,	"Mercedes-Benz-C300":173,	"Mercedes-Benz-C350":174,	"Mercedes-Benz-CLA200":175,	"Mercedes-Benz-CLA250":176,	"Mercedes-Benz-CLK200":177,	"Mercedes-Benz-CLS350":178,	"Mercedes-Benz-Citan":179,	"Mercedes-Benz-E200":180,	"Mercedes-Benz-E220":181,	"Mercedes-Benz-E240":182,	"Mercedes-Benz-E250":183,	"Mercedes-Benz-E300":184,	"Mercedes-Benz-E320":185,	"Mercedes-Benz-E350":186,	"Mercedes-Benz-EQC400":187,	"Mercedes-Benz-EQS":188,	"Mercedes-Benz-GLA200":189,	"Mercedes-Benz-GLA250":190,	"Mercedes-Benz-GLA45AMG":191,	"Mercedes-Benz-GLC220":192,	"Mercedes-Benz-GLC250":193,	"Mercedes-Benz-GLC300":194,	"Mercedes-Benz-GLC350":195,	"Mercedes-Benz-GLE350":196,	"Mercedes-Benz-ML320":197,	"Mercedes-Benz-ML350":198,	"Mercedes-Benz-Others":199,	"Mercedes-Benz-S350":200,	"Mercedes-Benz-SL500":201,	"Mercedes-Benz-SLK200":202,	"Mercedes-Benz-Sprinter":203,	"Mercedes-Benz-Vito":204,	"Opel-Agila":205,	"Opel-Ampera":206,	"Opel-Combo":210,	"Opel-Corsa":211,	"Opel-GrandlandX":212,	"Opel-Insignia":213,	"Opel-Meriva":214,	"Opel-Mokka":215,	"Opel-MokkaX":216,	"Opel-Tigra":218,	"Opel-Vectra":219,	"Peugeot-2008":221,	"Peugeot-206":222,	"Peugeot-3008":225,	"Peugeot-307":226,	"Peugeot-308":227,	"Peugeot-508":230,	"Peugeot-Expert":231,	"Peugeot-Partner":232,	"Peugeot-RCZ":233,	"Renault-Arkana":234,	"Renault-Clio":236,	"Renault-Espace":237,	"Renault-GrandScenic":238,	"Renault-Kangoo":239,	"Renault-Laguna":240,	"Renault-Master":241,	"Renault-Megane":242,	"Renault-Modus":243,	"Renault-Scenic":244,	"Renault-Trafic":246,	"Skoda-Citigo":249,	"Skoda-Enyaq":250,	"Skoda-Fabia":251,	"Skoda-Karoq":253,	"Skoda-Kodiaq":254,	"Skoda-Octavia":255,	"Tesla-ModelS":260,	"Tesla-ModelX":261,	"Toyota-Auris":262,	"Toyota-Aygo":264,	"Toyota-C-HR":265,	"Toyota-Camry":266,	"Toyota-Corolla":267,	"Toyota-CorollaVerso":268,	"Toyota-Hilux":269,	"Toyota-LandCruiser":270,"Toyota-RAV4":273,"Toyota-Verso-S":275,"Toyota-Yaris":276,"Volkswagen-Amarok":277,	"Volkswagen-Arteon":278,	"Volkswagen-Caddy":279,	"Volkswagen-Eos":280,	"Volkswagen-Golf":281,	"Volkswagen-GolfCabriolet":282,	"Volkswagen-GolfGTE":283,	"Volkswagen-GolfGTI":284,	"Volkswagen-GolfPlus":285,	"Volkswagen-GolfSportsvan":286,	"Volkswagen-GolfVariant":287,	"Volkswagen-ID.3":288,	"Volkswagen-ID.4":289,	"Volkswagen-Jetta":290,	"Volkswagen-Others":291,	"Volkswagen-Passat":292,	"Volkswagen-PassatCC":293,	"Volkswagen-PassatVariant":294,	"Volkswagen-Polo":295,	"Volkswagen-Scirocco":296,	"Volkswagen-Sharan":297,	"Volkswagen-T-Roc":298,	"Volkswagen-T5Transporter":299,	"Volkswagen-T6Transporter":300,	"Volkswagen-Tiguan":301,	"Volkswagen-Touareg":302,	"Volkswagen-Touran":303,	"Volkswagen-Transporter":304,	"Volkswagen-e-Golf":305,	"Volkswagen-up":306,	"Volvo-C30":307,	"Volvo-C70":308,	"Volvo-Others":309,	"Volvo-S40":310,	"Volvo-S60":311,	"Volvo-V40":314,	"Volvo-V40CrossCountry":315,	"Volvo-V60":317,	"Volvo-V70":319,"Volvo-XC40":321,"Volvo-XC70":323,"Volvo-XC90":324}
        m=st.selectbox(f"Make & Model ",["Audi-A1",	"Audi-A3","Audi-A4","Audi-A4allroad","Audi-A5","Audi-A6","Audi-A6allroad",	"Audi-A7",	"Audi-A8",	"Audi-Others",	"Audi-Q2",	"Audi-Q3",	"Audi-Q4e-tron",	"Audi-Q5",	"Audi-Q7",	"Audi-Q8",	"Audi-R8",	"Audi-S5",	"Audi-SQ5",	"Audi-SQ7",	"Audi-TT",	"Audi-e-tron",	"Audi-e-tronGT",	"BMW-116",	"BMW-118",	"BMW-120",	"BMW-218",	"BMW-220",	"BMW-225",	"BMW-316",	"BMW-318",	"BMW-320",	"BMW-325",	"BMW-328",	"BMW-330",	"BMW-335",	"BMW-420",	"BMW-428",	"BMW-435",	"BMW-520",	"BMW-523",	"BMW-525",	"BMW-530",	"BMW-535",	"BMW-640",	"BMW-650",	"BMW-730",	"BMW-740",	"BMW-M3",	"BMW-M4",	"BMW-Others",	"BMW-X1",	"BMW-X2",	"BMW-X3",	"BMW-X5",	"BMW-X6",	"BMW-Z3",	"BMW-Z4",	"BMW-i3",	"BMW-i4",	"BMW-iX",	"BMW-iX3",	"Chevrolet-Aveo",	"Chevrolet-Camaro",	"Chevrolet-Captiva",	"Chevrolet-Corvette",	"Chevrolet-Cruze",	"Citroen-Berlingo",	"Citroen-C2",	"Citroen-C3Picasso",	"Citroen-C4",	"Citroen-C4Picasso",	"Citroen-C5",	"Citroen-DS3",	"Citroen-GrandC4Picasso",	"Citroen-Jumpy",	"Citroen-Others",	"Citroen-XsaraPicasso",	"Dacia-Duster",	"Dacia-Logan",	"Fiat-500",	"Fiat-500C",	"Fiat-500L",	"Fiat-Doblo",	"Fiat-PuntoEvo",	"Fiat-Stilo",	"Ford-C-Max",	"Ford-Explorer",	"Ford-F150",	"Ford-Fiesta",	"Ford-Focus",	"Ford-FocusC-Max",	"Ford-FocusCC",	"Ford-Fusion",	"Ford-Galaxy",	"Ford-Ka/Ka+",	"Ford-Kuga",	"Ford-Mondeo",	"Ford-Mustang",	"Ford-MustangMach-E",	"Ford-Puma",	"Ford-Ranger",	"Ford-S-Max",	"Ford-Transit",	"Ford-TransitConnect",	"Ford-TransitCustom",	"Honda-Accord",	"Honda-CR-V",	"Honda-Civic",	"Honda-Jazz",	"Hyundai-Getz",	"Hyundai-Ioniq5",	"Hyundai-Kona",	"Hyundai-Nexo",	"Hyundai-i10",	"Hyundai-i20",	"Hyundai-i30",	"Hyundai-iX35",	"Kia-CeedSW/ceedSW",	"Kia-Ceed",	"Kia-EV6",	"Kia-Optima",	"Kia-Picanto",	"Kia-Sorento",	"Kia-Soul",	"Kia-Sportage",	"Kia-Venga",	"Kia-XCeed",	"Mazda-2",	"Mazda-5",	"Mazda-6",	"Mazda-CX-3",	"Mazda-CX-30",	"Mercedes-Benz-A150",	"Mercedes-Benz-A160",	"Mercedes-Benz-A170",	"Mercedes-Benz-A200",	"Mercedes-Benz-A250",	"Mercedes-Benz-A45AMG",	"Mercedes-Benz-B170",	"Mercedes-Benz-B180",	"Mercedes-Benz-B200",	"Mercedes-Benz-B250",	"Mercedes-Benz-C180",	"Mercedes-Benz-C200",	"Mercedes-Benz-C220",	"Mercedes-Benz-C250",	"Mercedes-Benz-C300",	"Mercedes-Benz-C350",	"Mercedes-Benz-CLA200",	"Mercedes-Benz-CLA250",	"Mercedes-Benz-CLK200",	"Mercedes-Benz-CLS350",	"Mercedes-Benz-Citan",	"Mercedes-Benz-E200",	"Mercedes-Benz-E220",	"Mercedes-Benz-E240",	"Mercedes-Benz-E250",	"Mercedes-Benz-E300",	"Mercedes-Benz-E320",	"Mercedes-Benz-E350",	"Mercedes-Benz-EQC400",	"Mercedes-Benz-EQS",	"Mercedes-Benz-GLA200",	"Mercedes-Benz-GLA250",	"Mercedes-Benz-GLA45AMG",	"Mercedes-Benz-GLC220",	"Mercedes-Benz-GLC250",	"Mercedes-Benz-GLC300",	"Mercedes-Benz-GLC350",	"Mercedes-Benz-GLE350",	"Mercedes-Benz-ML320",	"Mercedes-Benz-ML350",	"Mercedes-Benz-Others",	"Mercedes-Benz-S350",	"Mercedes-Benz-SL500",	"Mercedes-Benz-SLK200",	"Mercedes-Benz-Sprinter",	"Mercedes-Benz-Vito",	"Opel-Agila",	"Opel-Ampera",	"Opel-Combo",	"Opel-Corsa",	"Opel-GrandlandX",	"Opel-Insignia",	"Opel-Meriva",	"Opel-Mokka",	"Opel-MokkaX",	"Opel-Tigra",	"Opel-Vectra",	"Peugeot-2008",	"Peugeot-206",	"Peugeot-3008",	"Peugeot-307",	"Peugeot-308",	"Peugeot-508",	"Peugeot-Expert",	"Peugeot-Partner",	"Peugeot-RCZ",	"Renault-Arkana",	"Renault-Clio",	"Renault-Espace",	"Renault-GrandScenic",	"Renault-Kangoo",	"Renault-Laguna",	"Renault-Master",	"Renault-Megane",	"Renault-Modus",	"Renault-Scenic",	"Renault-Trafic",	"Skoda-Citigo",	"Skoda-Enyaq",	"Skoda-Fabia",	"Skoda-Karoq",	"Skoda-Kodiaq",	"Skoda-Octavia",	"Tesla-ModelS",	"Tesla-ModelX",	"Toyota-Auris",	"Toyota-Aygo",	"Toyota-C-HR",	"Toyota-Camry",	"Toyota-Corolla",	"Toyota-CorollaVerso",	"Toyota-Hilux",	"Toyota-LandCruiser",	"Toyota-RAV4",	"Toyota-Verso-S",	"Toyota-Yaris",	"Volkswagen-Amarok",	"Volkswagen-Arteon",	"Volkswagen-Caddy",	"Volkswagen-Eos",	"Volkswagen-Golf",	"Volkswagen-GolfCabriolet",	"Volkswagen-GolfGTE",	"Volkswagen-GolfGTI",	"Volkswagen-GolfPlus",	"Volkswagen-GolfSportsvan",	"Volkswagen-GolfVariant",	"Volkswagen-ID.3",	"Volkswagen-ID.4",	"Volkswagen-Jetta",	"Volkswagen-Others",	"Volkswagen-Passat",	"Volkswagen-PassatCC",	"Volkswagen-PassatVariant",	"Volkswagen-Polo",	"Volkswagen-Scirocco",	"Volkswagen-Sharan",	"Volkswagen-T-Roc",	"Volkswagen-T5Transporter",	"Volkswagen-T6Transporter",	"Volkswagen-Tiguan",	"Volkswagen-Touareg",	"Volkswagen-Touran",	"Volkswagen-Transporter",	"Volkswagen-e-Golf",	"Volkswagen-up",	"Volvo-C30",	"Volvo-C70",	"Volvo-Others",	"Volvo-S40",	"Volvo-S60",	"Volvo-V40","Volvo-V40CrossCountry","Volvo-V60","Volvo-V70","Volvo-XC40","Volvo-XC70","Volvo-XC90"])
        my_dict["make-model_index"]=make_model_dic[m]

        st.subheader("Age") 
        age = st.slider('Age of Auto ', min_value=0, max_value=27, value=8)        
        my_dict["age"]=age

        st.subheader("Empty Weight") 
        empty_weight = st.slider('Empty Weight of Auto ', 800, 6000, 2850)
        my_dict["empty_weight"]=empty_weight

        st.subheader("Power") 
        power = st.slider('Power of Auto ', 15, 300, 116)
        my_dict["power"]=power

        st.subheader("Fuel Consumption")
        comb = st.slider('Average fuel consumption per 100 km ', 3, 30, 10)
        my_dict["comb"]=comb

        st.subheader("Mileage")
        mileage = st.slider('Mileage of Auto ', 0, 900000, 500000)
        my_dict["mileage"]=mileage

        st.subheader("Engine Size")
        engine_size = st.slider('Engine_size of Auto' ,  0, 4000, 1980)
        my_dict["engine_size"]=engine_size

        st.subheader("Body Type") 
        body_type_dic= {"Off-Road_Pick-up": 1, "Compact": 2, "Station wagon": 3, "Van": 4, "Sedan": 5, "Convertible": 6, "Transporter": 7, "Coupe": 8, "Other": 9}
        b=st.selectbox(f"Body Type of Auto:", ['Off-Road_Pick-up','Compact','Station wagon','Van','Sedan','Convertible','Transporter','Coupe','Other'])
        my_dict["body_type_dic"]=body_type_dic[b]

        st.subheader("Fuel Type") 
        fuel_type_dic= {"Gasoline": 1, "Diesel": 2, "Hybrit": 3, "Electric": 4, "LPG": 5, "Others": 6}
        f=st.selectbox(f"Fuel Type of Auto:", ['Gasoline','Diesel','Hybrit','Electric','LPG','Others'])
        my_dict["fuel_type"]=fuel_type_dic[f]

        st.subheader("CO2 Emission")
        co2_emissions = st.slider('Co2 Emissions of Auto' ,  0, 200, 165)
        my_dict["co2_emissions"]=co2_emissions

        st.subheader("Emission Class") 
        emission_class_dic= {"Euro 1": 1, "Euro 2": 2, "Euro 3": 3, "Euro 4": 4, "Euro 5": 5, "Euro 6c": 6, "Euro 6d": 6, "Euro 6": 6}
        e=st.selectbox(f"Emission Class of Auto:", ['Euro 1','Euro 2','Euro 3','Euro 4','Euro 5','Euro 6c','Euro 6d','Euro 6'])
        my_dict["emission_class"]=emission_class_dic[e]

        st.subheader("Doors")
        doors = st.slider('Number of doors' , 1, 6, 4)
        my_dict["doors"]=doors

        st.subheader("Gear Box") 
        Gearbox_dic= {"Manual": 1, "Automatic": 2, "Semi-automatic": 3}
        g=st.selectbox(f"Gearbox Type", ['Manual','Automatic','Semi-automatic'])
        my_dict["Gearbox"]=Gearbox_dic[g]

        st.subheader("Colour") 
        colour_dic= {"grey": 1, "black": 2, "white": 3, "blue": 3, "red": 3, "other": 4}
        c=st.selectbox(f"colour of Auto:", ['grey','black','white','blue','red','other'])
        my_dict["colour"]=colour_dic[c]

        st.subheader("Upholstery") 
        upholstery_dic= {"Cloth": 1, "Fullleather": 2, "Partleather": 3, "Other": 4}
        u=st.selectbox(f"Upholstery Type", ['Cloth','Fullleather','Partleather','Other'])
        my_dict["upholstery"]=upholstery_dic[u]

        #drivetrain={'drivetrain 4WD','drivetrain Front','drivetrain Rear'}


        # comfort_convenience={'Parking assist system camera', 'Sliding door right', 'Multi-function steering wheel', 'Electrically heated windshield', 'Armrest', 'Air conditioning', 'Auxiliary heating', 'Keyless central door lock', 'Hill Holder', 'Automatic climate control 3 zones', 'Seat heating', 'Automatic climate control 4 zones', 'Park Distance Control', 'Parking assist system self-steering', 'Lumbar support', 'Split rear seats', 'Leather steering wheel', 'Wind deflector', 'Cruise control', 'Heated steering wheel', 'Parking assist system sensors front', 'Light sensor', 'Heads-up display', 'Air suspension', 'Leather seats', 'Parking assist system sensors rear', 'Sunroof', 'Electric tailgate', 'Fold flat passenger seat', 'Automatic climate control 2 zones', 'Seat ventilation', 'Automatic climate control', 'Rain sensor', 'Tinted windows', 'Electric backseat adjustment', 'Sliding door left', 'Panorama roof', 'Electrical side mirrors', 'Power windows', 'Navigation system', 'Start-stop system', 'Electrically adjustable seats', '360° camera', 'Massage seats'}
        # entertainment_media={'Digital cockpit', 'WLAN / WiFi hotspot', 'CD player', 'Apple CarPlay', 'Radio', 'On-board computer', 'Hands-free equipment', 'Integrated music streaming', 'Sound system', 'MP3', 'Induction charging for smartphones', 'Bluetooth', 'Television', 'USB', 'Android Auto', 'Digital radio'}
        # safety_security = {'Bi-Xenon headlights', 'Alarm system', 'Adaptive headlights', 'Side airbag', 'Tire pressure monitoring system', 'Speed limit control system', 'Distance warning system', 'Fog lights', 'Rear airbag', 'Power steering', 'LED Headlights', 'Immobilizer', 'ABS', 'Central door lock', 'Adaptive Cruise Control', 'High beam assist', 'Traffic sign recognition', 'Xenon headlights', 'Head airbag', 'Driver-side airbag', 'Full-LED headlights', 'Central door lock with remote control', 'Isofix', 'Night view assist', 'LED Daytime Running Lights', 'Driver drowsiness detection', 'Passenger-side airbag', 'Emergency brake assistant', 'Daytime running lights', 'Glare-free high beam headlights', 'Laser headlights', 'Lane departure warning system', 'Electronic stability control', 'Traction control', 'Blind spot monitor', 'Emergency system'}
        # extras = {'All season tyres', 'Handicapped enabled', 'Sliding door', 'Ski bag', 'Tuned car', 'Automatically dimming interior mirror', 'Emergency tyre', 'E10-enabled', 'Sport seats', 'Awning', 'Spoiler', 'Winter tyres', 'Emergency tyre repair kit', 'Shift paddles', 'Sport suspension', 'Range extender', 'Cargo barrier', 'Electronic parking brake', 'Sport package', 'Biodiesel conversion', 'Winter package', 'Voice Control', 'Steel wheels', 'Smokers package', 'Roof rack', 'Spare tyre', 'Right hand drive', 'Headlight washer system', 'Alloy wheels', 'Ambient lighting', 'Summer tyres', 'Double cabin', 'Touch screen', 'Trailer hitch', 'Catalytic Converter'}
        extras={'extras All season tyres','extras Alloy wheels','extras Alloy wheels 11','extras Alloy wheels 12','extras Alloy wheels 13','extras Alloy wheels 14','extras Alloy wheels 15','extras Alloy wheels 16','extras Alloy wheels 17','extras Alloy wheels 18','extras Alloy wheels 19','extras Alloy wheels 20','extras Alloy wheels 21',	'extras Alloy wheels 22',	'extras Alloy wheels 23',	'extras Alloy wheels 24',	'extras Alloy wheels 26',	'extras Ambient lighting',	'extras Automatically dimming interior mirror',	'extras Awning',	'extras Biodiesel conversion',	'extras Cargo barrier',	'extras Catalytic Converter',	'extras E10-enabled',	'extras Electronic parking brake',	'extras Emergency tyre',	'extras Emergency tyre repair kit',	'extras Handicapped enabled',	'extras Headlight washer system',	'extras Range extender',	'extras Right hand drive',	'extras Roof rack',	'extras Shift paddles',	'extras Ski bag',	'extras Sliding door',	'extras Smokers package',	'extras Spare tyre',	'extras Spoiler',	'extras Sport package',	'extras Sport seats',	'extras Sport suspension',	'extras Steel wheels',	}
        Comfort_Convenience={'Comfort_Convenience2_zones','Comfort_Convenience360°_camera','Comfort_Convenience3_zones','Comfort_Convenience4_zones','Comfort_ConvenienceAir_conditioning','Comfort_ConvenienceAir_suspension',	'Comfort_ConvenienceArmrest','Comfort_ConvenienceAutomatic_climate_control','Comfort_ConvenienceAuxiliary_heating',	'Comfort_ConvenienceCruise_control','Comfort_ConvenienceElectric_backseat_adjustment','Comfort_ConvenienceElectric_tailgate',	'Comfort_ConvenienceElectrical_side_mirrors',	'Comfort_ConvenienceElectrically_adjustable_seats',	'Comfort_ConvenienceElectrically_heated_windshield',	'Comfort_ConvenienceFold_flat_passenger_seat',	'Comfort_ConvenienceHeads_up_display',	'Comfort_ConvenienceHeated_steering_wheel',	'Comfort_ConvenienceHill_Holder',	'Comfort_ConvenienceKeyless_central_door_lock',	'Comfort_ConvenienceLeather_steering_wheel',	'Comfort_ConvenienceLight_sensor',	'Comfort_ConvenienceLumbar_support','Comfort_ConvenienceMassage_seats','Comfort_ConvenienceMulti_function_steering_wheel',	'Comfort_ConvenienceNavigation_system','Comfort_ConveniencePanorama_roof','Comfort_ConvenienceParking_assist_system_camera','Comfort_ConvenienceParking_assist_system_self_steering','Comfort_ConvenienceParking_assist_system_sensors_front','Comfort_ConvenienceParking_assist_system_sensors_rear',	'Comfort_ConveniencePower_windows','Comfort_ConvenienceRain_sensor','Comfort_ConvenienceSeat_heating',	'Comfort_ConvenienceSeat_ventilation',	'Comfort_ConvenienceSliding_door_left',	'Comfort_ConvenienceSliding_door_right','Comfort_ConvenienceSplit_rear_seats','Comfort_ConvenienceStart_stop_system','Comfort_ConvenienceSunroof','Comfort_ConvenienceTinted_windows','Comfort_ConvenienceWind_deflector'}	
        entertainment_media={'entertainment_media androidauto',	'entertainment_media applecarplay','entertainment_media bluetooth','entertainment_media cdplayer','entertainment_media digitalcockpit','entertainment_media digitalradio','entertainment_media hands-freeequipment','entertainment_media inductionchargingforsmartphones',	'entertainment_media integratedmusicstreaming',	'entertainment_media mp3','entertainment_media on-boardcomputer'}
        safety_security={'safety_security abs','safety_security adaptive_cruise_contro','safety_security adaptive_cruise_control','safety_security adaptive_headlights','safety_security alarm_system','safety_security bi-xenon_headlights','safety_security blind_spot_monitor','safety_security central_door_lock','safety_security central_door_lock_with_remote_control',	'safety_security daytime_running_lights',	'safety_security distance_warning_system',	'safety_security driver-side_airbag',	'safety_security driver_drowsiness_detection',	'safety_security electronic_stability_control',	'safety_security emergency_brake_assistant',	'safety_security emergency_system',	'safety_security fog_lights',	'safety_security full-led_headlights',	'safety_security glare-free_high_beam_headlights',	'safety_security head_airbag',	'safety_security high_beam_assist',	'safety_security immobilizer',	'safety_security isofix',	'safety_security led_daytime_running_lights',	'safety_security led_headlights',	'safety_security lane_departure_warning_system',	'safety_security laser_headlights',	'safety_security night_view_assist',	'safety_security passenger-side_airbag',	'safety_security power_steering',	'safety_security rear_airbag',	'safety_security side_airbag',	'safety_security speed_limit_control_system',	'safety_security tire_pressure_monitoring_system',	'safety_security traction_control',	'safety_security traffic_sign_recognition',	'safety_security xenon_headlights',}

        st.sidebar.subheader("EXTRAS")
        for i in extras:
                my_dict[i]=st.sidebar.checkbox(i)
        
        st.sidebar.subheader("CONFORT CONVENIENCE")
        for i in Comfort_Convenience:
                my_dict[i]=st.sidebar.checkbox(i)
        
        st.sidebar.subheader("ENTERTAINMENT MEDAIA")
        for i in entertainment_media:
                my_dict[i]=st.sidebar.checkbox(i)

        st.sidebar.subheader("SAFETY SECURITY")
        for i in safety_security:
                my_dict[i]=st.sidebar.checkbox(i)


        all_columns = ['drivetrain 4WD',
 'drivetrain Front',
 'drivetrain Rear',
 'Comfort_Convenience 2_zones',
 'Comfort_Convenience 360°_camera',
 'Comfort_Convenience 3_zones',
 'Comfort_Convenience 4_zones',
 'Comfort_Convenience Air_conditioning',
 'Comfort_Convenience Air_suspension',
 'Comfort_Convenience Armrest',
 'Comfort_Convenience Automatic_climate_control',
 'Comfort_Convenience Auxiliary_heating',
 'Comfort_Convenience Cruise_control',
 'Comfort_Convenience Electric_backseat_adjustment',
 'Comfort_Convenience Electric_tailgate',
 'Comfort_Convenience Electrical_side_mirrors',
 'Comfort_Convenience Electrically_adjustable_seats',
 'Comfort_Convenience Electrically_heated_windshield',
 'Comfort_Convenience Fold_flat_passenger_seat',
 'Comfort_Convenience Heads_up_display',
 'Comfort_Convenience Heated_steering_wheel',
 'Comfort_Convenience Hill_Holder',
 'Comfort_Convenience Keyless_central_door_lock',
 'Comfort_Convenience Leather_steering_wheel',
 'Comfort_Convenience Light_sensor',
 'Comfort_Convenience Lumbar_support',
 'Comfort_Convenience Massage_seats',
 'Comfort_Convenience Multi_function_steering_wheel',
 'Comfort_Convenience Navigation_system',
 'Comfort_Convenience Panorama_roof',
 'Comfort_Convenience Parking_assist_system_camera',
 'Comfort_Convenience Parking_assist_system_self_steering',
 'Comfort_Convenience Parking_assist_system_sensors_front',
 'Comfort_Convenience Parking_assist_system_sensors_rear',
 'Comfort_Convenience Power_windows',
 'Comfort_Convenience Rain_sensor',
 'Comfort_Convenience Seat_heating',
 'Comfort_Convenience Seat_ventilation',
 'Comfort_Convenience Sliding_door_left',
 'Comfort_Convenience Sliding_door_right',
 'Comfort_Convenience Split_rear_seats',
 'Comfort_Convenience Start_stop_system',
 'Comfort_Convenience Sunroof',
 'Comfort_Convenience Tinted_windows',
 'Comfort_Convenience Wind_deflector',
 'entertainment_media androidauto',
 'entertainment_media applecarplay',
 'entertainment_media bluetooth',
 'entertainment_media cdplayer',
 'entertainment_media digitalcockpit',
 'entertainment_media digitalradio',
 'entertainment_media hands-freeequipment',
 'entertainment_media inductionchargingforsmartphones',
 'entertainment_media integratedmusicstreaming',
 'entertainment_media mp3',
 'entertainment_media on-boardcomputer',
 'entertainment_media radio',
 'entertainment_media soundsystem',
 'entertainment_media television',
 'entertainment_media usb',
 'entertainment_media wlan/wifihotspot',
 'extras All season tyres',
 'extras Alloy wheels',
 'extras Alloy wheels 11',
 'extras Alloy wheels 12',
 'extras Alloy wheels 13',
 'extras Alloy wheels 14',
 'extras Alloy wheels 15',
 'extras Alloy wheels 16',
 'extras Alloy wheels 17',
 'extras Alloy wheels 18',
 'extras Alloy wheels 19',
 'extras Alloy wheels 20',
 'extras Alloy wheels 21',
 'extras Alloy wheels 22',
 'extras Alloy wheels 23',
 'extras Alloy wheels 24',
 'extras Alloy wheels 26',
 'extras Ambient lighting',
 'extras Automatically dimming interior mirror',
 'extras Awning',
 'extras Biodiesel conversion',
 'extras Cargo barrier',
 'extras Catalytic Converter',
 'extras E10-enabled',
 'extras Electronic parking brake',
 'extras Emergency tyre',
 'extras Emergency tyre repair kit',
 'extras Handicapped enabled',
 'extras Headlight washer system',
 'extras Range extender',
 'extras Right hand drive',
 'extras Roof rack',
 'extras Shift paddles',
 'extras Ski bag',
 'extras Sliding door',
 'extras Smokers package',
 'extras Spare tyre',
 'extras Spoiler',
 'extras Sport package',
 'extras Sport seats',
 'extras Sport suspension',
 'extras Steel wheels',
 'extras Summer tyres',
 'extras Touch screen',
 'extras Trailer hitch',
 'extras Tuned car',
 'extras Voice Contro',
 'extras Voice Control',
 'extras Winter package',
 'extras Winter tyres',
 'safety_security abs',
 'safety_security adaptive_cruise_contro',
 'safety_security adaptive_cruise_control',
 'safety_security adaptive_headlights',
 'safety_security alarm_system',
 'safety_security bi-xenon_headlights',
 'safety_security blind_spot_monitor',
 'safety_security central_door_lock',
 'safety_security central_door_lock_with_remote_control',
 'safety_security daytime_running_lights',
 'safety_security distance_warning_system',
 'safety_security driver-side_airbag',
 'safety_security driver_drowsiness_detection',
 'safety_security electronic_stability_control',
 'safety_security emergency_brake_assistant',
 'safety_security emergency_system',
 'safety_security fog_lights',
 'safety_security full-led_headlights',
 'safety_security glare-free_high_beam_headlights',
 'safety_security head_airbag',
 'safety_security high_beam_assist',
 'safety_security immobilizer',
 'safety_security isofix',
 'safety_security led_daytime_running_lights',
 'safety_security led_headlights',
 'safety_security lane_departure_warning_system',
 'safety_security laser_headlights',
 'safety_security night_view_assist',
 'safety_security passenger-side_airbag',
 'safety_security power_steering',
 'safety_security rear_airbag',
 'safety_security side_airbag',
 'safety_security speed_limit_control_system',
 'safety_security tire_pressure_monitoring_system',
 'safety_security traction_control',
 'safety_security traffic_sign_recognition',
 'safety_security xenon_headlights',
 'mileage',
 'fuel_type',
 'power',
 'seller',
 'body_type',
 'type',
 'seats',
 'doors',
 'warranty',
 'first_registration',
 'general_inspection',
 'full_service_history',
 'non_smoker_vehicle',
 'Gearbox',
 'engine_size',
 'gears',
 'cylinders',
 'empty_weight',
 'co2_emissions',
 'emission_class',
 'colour',
 'paint',
 'upholstery_colour',
 'upholstery',
 'age',
 'comb',
 'make-model_index']
        
        features_df = pd.DataFrame([my_dict]) 
        features_df = features_df.reindex(columns=all_columns, fill_value=0)
        
        #features_df = pd.DataFrame.from_dict([my_dict])
        #features_df = pd.get_dummies(features_df)
        #features_df = pd.get_dummies(features_df).reindex(columns=all_columns, fill_value=0)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        #Preprocess inputs

        #preprocess_df = features_df.copy()

        prediction = model.predict(features_df)
        #print("The Price : ",np.e**prediction[0])
        #np.e**prediction[0]

        if st.button('Predict'):

                st.success(f"The Prediction Price of the Car is €{int(np.e**prediction[0])}")


# else:
#     st.subheader("Dataset upload")
#     uploaded_file = st.file_uploader("Choose a file")
#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file,encoding= 'utf-8')
#         #Get overview of data
#         st.write(data.head())
#         st.markdown("<h3></h3>", unsafe_allow_html=True)
#         #Preprocess inputs
#         preprocess_df = preprocess(data, "Batch")
#         if st.button('Predict'):
#             #Get batch prediction
#             prediction = model.predict(preprocess_df)
#             prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
#             prediction_df = prediction_df.replace({1:'Yes, the passenger survive.', 0:'No, the passenger died'})

#             st.markdown("<h3></h3>", unsafe_allow_html=True)
#             st.subheader('Prediction')
#             st.write(prediction_df) 

if __name__ == '__main__':
        main()



