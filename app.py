import streamlit as st

st.title('Voting Eligibility Checker')

age = st.number_input('Enter your age:', min_value=0, max_value=120, value=18)

if st.button('Check Eligibility'):
    if age >= 18:
        st.success('You are eligible to vote!')
    else:
        st.error('You are not eligible to vote yet.')
