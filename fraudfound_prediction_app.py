import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import base64

#============ Set Page Title and Icon ===========

st.set_page_config(
    page_title="Vehicle Insurance Claim Fraud Detection",
    page_icon="🚗"
)

#============= Set Page Background =============

# ====== Function for Local Background ======
def local_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"url('data:image/png;base64,{encoded}')"

bg_image = local_bg("pexels-albinberlin-919073.jpg")  # Replace with your local image file

# ====== CSS for Gradient Overlay + Styling ======
st.markdown(
    f"""
    <style>
    @keyframes gradientShift {{
        0% {{background-position: 0% 50%;}}
        50% {{background-position: 100% 50%;}}
        100% {{background-position: 0% 50%;}}
    }}

    .stApp {{
        background: 
            linear-gradient(120deg, rgba(135,206,235,0.05), rgba(135,206,235,0.05), rgba(135,206,235,0.05)),
            {bg_image};
        background-size: 400% 400%, cover;
        background-repeat: no-repeat;
        background-position: center;
        animation: gradientShift 15s ease infinite;
    }}

    /* Header كبير */
    h1 {{
        color: white !important;
        font-size: 38px !important;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }}

    /* باقي النصوص باللون الأبيض الافتراضي وحجم طبيعي */
    h2, h3, h4, h5, h6, p, div, span {{
        color: white !important;
        font-size: 18px !important;
        font-weight: 500;
    }}

    .stButton button {{
        color: white !important;
        background-color: #007BFF;
        border-radius: 10px;
        font-weight: bold;
        padding: 12px 25px;
        font-size: 18px;
    }}
    .stButton button:hover {{
        background-color: #0056b3;
        border: 1px solid white;
    }}

    .result-box {{
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white !important;
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#============ Handling Color In Page ===========

st.markdown(
    """
    <style>
    /* زرار أصغر وأبعاد أقل */
    .stButton button {
        color: white !important;
        background-color: #007BFF;  /* لون الزرار */
        border-radius: 8px;
        font-weight: bold;
        padding: 8px 18px;  /* أقل من قبل */
        font-size: 16px;    /* أصغر */
    }
    .stButton button:hover {
        background-color: #0056b3;
        border: 1px solid white;
    }

    /* صندوق الرسالة لون لبني */
    .result-box {
        background-color: rgba(173, 216, 230, 0.7);  /* لون لبني شفاف */
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: black !important;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#===========Load Model ========================
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
x_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    random_state=42,
    max_depth=30,
    min_samples_split=4,
    min_samples_leaf=3 ,
    n_estimators=100
)

rf.fit(x_train, y_train)

# ========== Dics Of Columns ===============

Dayofweek = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}
make = {'Accura': 0, 'BMW': 1, 'Chevrolet': 2, 'Dodge': 3, 'Ferrari': 4, 'Ford': 5, 'Honda': 6, 'Jaguar': 7, 'Lexus': 8, 'Mazda': 9, 'Mercedes': 10, 'Mercury': 11, 'Nisson': 12, 'Pontiac': 13, 'Porche': 14, 'Saab': 15, 'Saturn': 16, 'Toyota': 17, 'VW': 18}
accidentarea = {'Rural': 0, 'Urban': 1}
dayofweekclaimed = {'0': 0, 'Friday': 1, 'Monday': 2, 'Saturday': 3, 'Sunday': 4, 'Thursday': 5, 'Tuesday': 6, 'Wednesday': 7}
sex = {'Female': 0, 'Male': 1}
maritalstatus = {'Divorced': 0, 'Married': 1, 'Single': 2, 'Widow': 3}
fault = {'Policy Holder': 0, 'Third Party': 1}
vehiclecategory = {'Sedan': 0, 'Sport': 1, 'Utility': 2}
vehicleprice = {'20000 to 29000': 0, '30000 to 39000': 1, '40000 to 59000': 2, '60000 to 69000': 3, 'less than 20000': 4, 'more than 69000': 5}
days_policy_accident = {'1 to 7': 0, '15 to 30': 1, '8 to 15': 2, 'more than 30': 3, 'none': 4}
days_policy_claim = {'15 to 30': 0, '8 to 15': 1, 'more than 30': 2, 'none': 3}
pastnumberofclaims = {'1': 0, '2 to 4': 1, 'more than 4': 2, 'none': 3}
ageofvehicle = {'2 years': 0, '3 years': 1, '4 years': 2, '5 years': 3, '6 years': 4, '7 years': 5, 'more than 7': 6, 'new': 7}
ageofpolicyholder = {'16 to 17': 0, '18 to 20': 1, '21 to 25': 2, '26 to 30': 3, '31 to 35': 4, '36 to 40': 5, '41 to 50': 6, '51 to 65': 7, 'over 65': 8}
policereportfiled = {'No': 0, 'Yes': 1}
witnesspresent = {'No': 0, 'Yes': 1}
agenttype = {'External': 0, 'Internal': 1}
numberofsuppliments = {'1 to 2': 0, '3 to 5': 1, 'more than 5': 2, 'none': 3}
addresschange_claim = {'1 year': 0, '2 to 3 years': 1, '4 to 8 years': 2, 'no change': 3, 'under 6 months': 4}
numberofcars = {'1 vehicle': 0, '2 vehicles': 1, '3 to 4': 2, '5 to 8': 3, 'more than 8': 4}
policytype = {'Sedan - All Perils': 0, 'Sedan - Collision': 1, 'Sedan - Liability': 2, 'Sport - All Perils': 3, 'Sport - Collision': 4, 'Sport - Liability': 5, 'Utility - All Perils': 6, 'Utility - Collision': 7, 'Utility - Liability': 8}
basepolicy = {'All Perils': 0, 'Collision': 1, 'Liability': 2}

# ========= Set Title ==========

st.markdown(
    "<h1 style='text-align: center; font-size: 56px; font-weight: bold; color: white;'>🚗 Vehicle Insurance Claim Fraud Detection</h1>", 
    unsafe_allow_html=True
)

# ======= Take Input Value To Predict =======

s1 = st.number_input('Week Of Month :' , min_value=1, max_value=5)
S1 = s1

s2 = st.selectbox('Day Of Week :', options=list(Dayofweek.keys()))
S2 = Dayofweek[s2]

s3= st.selectbox('Make :', options=list(make.keys()))
S3 = make[s3]

s4 = st.selectbox('Accident Area :', options=list(accidentarea.keys()))
S4 = accidentarea[s4]

s5 = st.selectbox('Day Of Week Claimed :', options=list(dayofweekclaimed.keys()))
S5 = dayofweekclaimed[s5]

s6 = st.number_input('Week Of Month Claimed :' , min_value=1, max_value=5)
S6 = s6

s7 = st.selectbox('Sex :', options=list(sex.keys()))
S7 = sex[s7]

s8 = st.selectbox('marital Status :', options=list(maritalstatus.keys()))
S8 = maritalstatus[s8]

s9 = st.number_input('Age :' , min_value=0, max_value=100)
S9 = s9

s10 = st.selectbox('Fault :', options=list(fault.keys()))
S10 = fault[s10]

s11 = st.selectbox('policy Type :', options=list(policytype.keys()))
S11 = policytype[s11]

s12 = st.selectbox('Vehicle Category :', options=list(vehiclecategory.keys()))
S12 = vehiclecategory[s12]

s13 = st.selectbox('Vehicle Price :', options=list(vehicleprice.keys()))
S13 = vehicleprice[s13]

s14 = st.number_input('Policy Number :' , min_value=1, max_value=15420)
S14 = s14

s15 = st.number_input('RepNumber : ', min_value=1, max_value=100)
S15 = s15

s16 = st.number_input('Deductible (100 , 200 , 300 , .....) : ', min_value=0, max_value=10000)
S16 = s16

s17 = st.number_input('Driver Rating : ', min_value=1, max_value=4)
S17 = s17

st18 = st.selectbox('Days Policy Accident :', options=list(days_policy_accident.keys()))
S18 = days_policy_accident[st18]

s19 = st.selectbox('Days Policy Claim :', options=list(days_policy_claim.keys()))
S19 = days_policy_claim[s19]

s20 = st.selectbox('Past Number Of Claims :', options=list(pastnumberofclaims.keys()))
S20 = pastnumberofclaims[s20]

s21 = st.selectbox('Age Of Vehicle :', options=list(ageofvehicle.keys()))
S21 = ageofvehicle[s21]

s22 = st.selectbox('Age Of Policy Holder :', options=list(ageofpolicyholder.keys()))
S22 = ageofpolicyholder[s22]

s23 = st.selectbox('Police Report Filed :', options=list(policereportfiled.keys()))
S23 = policereportfiled[s23]

s24 = st.selectbox('Witness Present :', options=list(witnesspresent.keys()))
S24 = witnesspresent[s24]

s25 = st.selectbox('Agent Type :', options=list(agenttype.keys()))
S25 = agenttype[s25]

s26 = st.selectbox('Number Of Suppliments :', options=list(numberofsuppliments.keys()))
S26 = numberofsuppliments[s26]

s27 = st.selectbox('Address Change Claim :', options=list(addresschange_claim.keys()))
S27 = addresschange_claim[s27]

s28 = st.selectbox('Number Of Cars :', options=list(numberofcars.keys()))
S28 = numberofcars[s28]

s29 = st.number_input('Year :', min_value=1990, max_value=2023)
S29 = s29

s30 = st.selectbox('Base Policy :', options=list(basepolicy.keys()))
S30 = basepolicy[s30]

#============ 🟢 Create DataFrame for Input ==================
 
input_data = []
encode_data = []

for i in range(1, 31):
    if i == 1:
        input_data.append(s1)
        encode_data.append(S1)
    elif i == 2:
        input_data.append(s2)
        encode_data.append(S2)
    elif i == 3:
        input_data.append(s3)
        encode_data.append(S3)
    elif i == 4:
        input_data.append(s4)
        encode_data.append(S4)
    elif i == 5:
        input_data.append(s5)
        encode_data.append(S5)
    elif i == 6:
        input_data.append(s6)
        encode_data.append(S6)
    elif i == 7:
        input_data.append(s7)
        encode_data.append(S7)
    elif i == 8:
        input_data.append(s8)
        encode_data.append(S8)
    elif i == 9:
        input_data.append(s9)
        encode_data.append(S9)
    elif i == 10:
        input_data.append(s10)
        encode_data.append(S10)
    elif i == 11:
        input_data.append(s11)
        encode_data.append(S11)
    elif i == 12:
        input_data.append(s12)
        encode_data.append(S12)
    elif i == 13:
        input_data.append(s13)
        encode_data.append(S13)
    elif i == 14:
        input_data.append(s14)
        encode_data.append(S14)
    elif i == 15:
        input_data.append(s15)
        encode_data.append(S15)
    elif i == 16:
        input_data.append(s16)
        encode_data.append(S16)
    elif i == 17:
        input_data.append(s17)
        encode_data.append(S17)
    elif i == 18:
        input_data.append(S18)
        encode_data.append(S18)
    elif i == 19:
        input_data.append(s19)
        encode_data.append(S19)
    elif i == 20:
        input_data.append(s20)
        encode_data.append(S20)
    elif i == 21:
        input_data.append(s21)
        encode_data.append(S21)
    elif i == 22:
        input_data.append(s22)
        encode_data.append(S22)
    elif i == 23:
        input_data.append(s23)
        encode_data.append(S23)
    elif i == 24:
        input_data.append(s24)
        encode_data.append(S24)
    elif i == 25:
        input_data.append(s25)
        encode_data.append(S25)
    elif i == 26:
        input_data.append(s26)
        encode_data.append(S26)
    elif i == 27:
        input_data.append(s27)
        encode_data.append(S27)
    elif i == 28:
        input_data.append(s28)
        encode_data.append(S28)
    elif i == 29:
        input_data.append(s29)
        encode_data.append(S29)
    elif i == 30:
        input_data.append(s30)
        encode_data.append(S30)

#============ Create DataFrame for Input ============

df_input = pd.DataFrame([input_data], columns=rf.feature_names_in_)
df_encoded = pd.DataFrame([encode_data], columns=rf.feature_names_in_)

#========= 🟢 Display Inputs =======================

st.subheader("🔹 Original Input")
st.write(df_input)

st.subheader("🔹 Encoded Input")
st.write(df_encoded)

#============== Predict =============================

result = rf.predict(df_encoded)

col1, col2, col3 = st.columns([1,0.5,1])
with col2:
    predict_btn = st.button("🔍 Predict ")

# ============== Show Result ========================

if predict_btn:
    st.balloons()  # 🎈 Balloons effect

    if result == 0:
        st.markdown(
            "<div class='result-box'>المطالبة ليست احتيالية (Claim Legit) ✅</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,10,1])
        with col2:
            st.image("1.png", width=900)

    elif result == 1:
        st.markdown(
            "<div class='result-box'>المطالبة احتيالية (Fraudulent Claim) ⚠️</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,10,1])
        with col2:
            st.image("2.png", width=1200)
