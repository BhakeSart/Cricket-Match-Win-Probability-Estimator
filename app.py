import streamlit as st
import pandas as pd
import pickle

# teams
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

# venues
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']


pipe = pickle.load(open("pipe.pkl", "rb"))
st.title("Cricket Match Win Probability Estimator ~ By Sarthak Bhake")

col1, col2 = st.columns(2)
with col1:
    batting = st.selectbox("Select the Batting Team", sorted(teams))


with col2:
    bowling = st.selectbox("Select the Bowling Team", sorted(teams))

city = st.selectbox(
    "Select the city where the match is being played", sorted(cities))
target = st.number_input("Target")
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input("Current Score")
with col4:
    overs = st.number_input("Overs Completed")
with col5:
    wickets = st.number_input("Wickets Fallen")

if st.button("Predict Probability"):
    runs_left = target-score
    balls_left = 120-(overs*6)
    wickets = 10-wickets
    currentrr = score/overs
    requiredrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting], 'bowling_team': [bowling], 'city': [city], 'runs_left': [runs_left], 'balls_left': [
                            balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'cur_run_rate': [currentrr], 'req_run_rate': [requiredrr]})

    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    st.header(batting+"- "+str(round(winprob*100))+"%")

    st.header(bowling+"- "+str(round(lossprob*100))+"%")
