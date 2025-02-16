import streamlit as st
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os


def clean_data():
    df = pd.read_csv("Data/diabetes.csv")
    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace zero values with the mean of the respective column
    for column in columns_with_zero:
        df[column].replace(0, df[column].mean(), inplace=True)

    return df



def number_inputs():
    data = clean_data()
    inputs = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

    input_dict = {}
    with st.container():
        st.markdown("<div class='inputs'>",unsafe_allow_html=True)
        for feature in inputs:
            input_dict[feature] = st.number_input(label=feature,
                            min_value=float(0),
                            max_value=float(data[feature].max()),
                            placeholder="Enter a Number")

        st.markdown("</div>",unsafe_allow_html=True)
    return input_dict

def scaled_values(input_dict):
    data = clean_data()

    X = data.drop("Outcome",axis=1)
    scaled_dict = {}

    for key,value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value =  (value - min_val) / (max_val - min_val)
        
        scaled_dict[key] = scaled_value

    return scaled_dict

def generate_graph(input_data):
    input_data = scaled_values(input_data)

    categories = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r = [input_data[feature] for feature in categories],
        theta=categories,
        fill='toself',
        name='Features'
    ))
    fig.update_layout(
        polar = dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )),
            showlegend=True,
            
    )
    return fig

def add_predictions(input_data):
    model = pickle.load(open("Model/model.pkl","rb"))
    sc = pickle.load(open("Model/scaler.pkl","rb"))

    input_array = np.array(list(input_data.values())).reshape(1,-1)
    scaled_input_array = sc.transform(input_array)

    prediction = model.predict(scaled_input_array)

    if prediction[0] == 0:
        st.markdown(f"""<div class='container2'>
                    <h2> Cell Cluster predictions </h2>
                    <p>The result is : </p>
                    <p id='resultB'> Benign</p>
                    <p>Probability of being benign : <span id='benign'>{model.predict_proba(scaled_input_array)[0][0]} </span> </p>
                    <p>Probability of being Malicious : <span id='malicious'>{model.predict_proba(scaled_input_array)[0][1]} </span> </p>
                    <p>This app can assist medical professionals in making a diagnosis, but should not be used as a substitude for a professional diagnosis.</p>
                </div>""",unsafe_allow_html=True)
    else:
       
        st.markdown(f"""<div class='container2'>
                    <h2> Cell Cluster predictions </h2>
                    <p>The result is : </p>
                    <p id='resultM'>Malicious</p>
                    <p>Probability of being benign : <span id='benign'>{model.predict_proba(scaled_input_array)[0][0]} </span> </p>
                    <p>Probability of being Malicious : <span id='malicious'>{model.predict_proba(scaled_input_array)[0][1]} </span> </p>
                    <p>This app can assist medical professionals in making a diagnosis, but should not be used as a substitude for a professional diagnosis.</p>
                </div>""",unsafe_allow_html=True)
def main():
    st.set_page_config(
        page_title="Diabetes Prediction's App",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Read image and encode it in base64
    image_path = "Images/image.png"
    try:
        with open(image_path,"rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        st.error(f"Image not found at : {os.path.abspath(image_path)}")
    custom_css = """
            <style>
            
                h1{
                    text-align:center;
                    color:#61bdbd;
                    transition: transform 0.6s ease-out,filter 0.8s ease-out;
                }
                h1:hover{
                        transform:scale(1.1);
                        filter:drop-shadow(0px 0px 7px #fff);
                }
            
                /* for image */
                #diabetes-image{
                    width:100%;
                    max-width:500px;
                    border-radius:13px;
                    box-shadow:0 4px 8px #d0a28e;
                    display:block;
                    margin: 0 auto;
                    position:relative;
                    top:10px;
                    transition: transform 0.7s ease-out, filter 0.7s ease-out;
                }
                #diabetes-image:hover{
                    transform: scale(1.05);
                    filter:drop-shadow(0px 0px 9px #d0a28e);
                }
                p{
                    margin-top:20px;
                    font-size:16px;
                }
                .inputs{
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    width:200px;
                }
            
                
                .container2{
            background: linear-gradient(to left,#27426d,white);
            padding:15px 20px;
            border-radius: 18px;
            width:290px;
            transition: transform 0.3s ease-in-out,filter 0.3s ease-in;
           
            }
        .container2 h2{
            text-align:center;
            color:#000;
          
        }
        .container2 p{
                color:#000;
                font-weight:600;
                
            }

        .container2:hover{
            transform: scale(1.05);
            filter: drop-shadow(0px 0px 14px #fff);
           
        }
        #benign{
            color: rgb(0,210,0);
            padding: 3px 9px;
            background-color: #000;
            border-radius: 9px;
        }
        #malicious{
            color: rgb(210,0,0);
            padding: 3px 9px;
            background-color: #000;
            border-radius: 9px;
        }

        #resultM{
            background-color: rgb(210,0,0);
            padding:3px 6px;
            color:#fff;
            font-weight:600;
            width:120px;
            border-radius:19px;
            text-align:center;
            margin-left:64px;
        }

        #resultB{
                background-color: rgb(0,210,0);
            padding:3px 6px;
            color:#fff;
            font-weight:600;
            width:120px;
            border-radius:19px;
            text-align:center;
            margin-left:64px;
            }
            </style>
    """

    with st.container():
        st.markdown("<h1>Diabetes Prediction's App</h1>",unsafe_allow_html=True)

        st.markdown(f'<img src="data:image/png;base64,{encoded_image}" alt="Diabetes Image" id="diabetes-image">',unsafe_allow_html=True)
        st.markdown(custom_css,unsafe_allow_html=True)

        st.markdown("""<p id='diabetes-description'>
                    Diabetes is a chronic medical condition that occurs
                    when the body is unable to properly regulate blood sugar (glucose) levels.
                    This happens either because the pancreas does not produce enough insulin (Type 1 diabetes)
                    or because the body's cells do not respond properly to insulin (Type 2 diabetes).
                    High blood sugar levels over time can lead to serious health complications, 
                    including heart disease, nerve damage, kidney failure, and vision loss.
                    Common symptoms of diabetes include excessive thirst, frequent urination, fatigue, blurred vision, and slow healing of wounds.
                    Early diagnosis and proper management, including lifestyle changes, medication, and monitoring, are essential for preventing complications.
                </p>
        """,unsafe_allow_html=True)

        st.markdown("""<p>
                    This Streamlit-based web application is designed to predict whether a person has diabetes
                     using machine learning. The app allows users to input health-related data such as
                     age, blood pressure, BMI, glucose levels, and insulin levels. Based on these inputs, 
                    the trained machine learning model analyzes the data and provides a prediction.
                    </p>""",unsafe_allow_html=True)
        
    
        input_dict = number_inputs()

    col1,col2 = st.columns([3,2])

    with col1:
            radar = generate_graph(input_dict)
            st.plotly_chart(radar)

    with col2:
            add_predictions(input_dict)
 

if __name__ == "__main__":
    main()