# import relevant packages
import streamlit as st
from streamlit.hashing import _CodeHasher
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math

st.set_page_config(
    page_title="Self Driving Car ND", # String or None. Strings get appended with "• Streamlit". 
    page_icon=":blue_car:", # String, anything supported by st.image, or None.
    layout="centered", # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="expanded", # Can be "auto", "expanded", "collapsed"
)

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


def main():
    state = _get_state()
    pages = {
        "Home": project_explanation,
        "Experiment": parameter_experiment,
        "Hough Lines": hough_lines,
        "Extrapolate Lines": extrapolate_lines,
        "Stabilize Lines": stabilize_lines,
        "Fill Lines": fill_lines
    }

    st.sidebar.title(":bookmark_tabs: Navigation")
    
    # Display the selected page with the session state
    # if state.clicked1 or state.clicked2 or state.clicked3:
        # page = "Your Strategy"
    # else:
        # page = st.sidebar.radio("Select your page", tuple(pages.keys()),)

    page = st.sidebar.radio("Select your page", tuple(pages.keys()),)
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def project_explanation(state):
    st.title("Finding Lane Lines on the Road")
    # st.write('----')
    st.markdown("[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)")

    st.markdown('`Welcome to the data-driven application`')
    st.markdown('<p style="text-align: justify;">When we drive, we use our eyes to decide where to go. \
                 The lines on the road that show us where the lanes are act as our constant reference \
                 for where to steer the vehicle. Naturally, one of the first things we would like to do \
                 in developing a self-driving car is to automatically detect lane lines using an algorithm. \
                 In this project you will detect lane lines in images using Python and OpenCV. OpenCV means \
                 "Open-Source Computer Vision", which is a package that has many useful tools for analyzing \
                 images.</p>', unsafe_allow_html=True)

    col5, col6 = st.beta_columns((1,1.5))
    col5.write('----')

    col3, col4 = st.beta_columns((1,1.4))
    col4.markdown('<p style="text-align: justify;">To complete the project, two files will be submitted: \
                   a file containing project code and a file containing a brief write up explaining your \
                   solution. We have included template files to be used both for the code and the writeup. \
                   The code file is called P1.ipynb and the writeup template is writeup_template.md</p>', unsafe_allow_html=True)
    col4.markdown("`Go to the Get Started page to enter details ->`")
    img2 = Image.open("examples/laneLines_thirdPass.jpg")
    col3.image(img2, width=250)

    col7, col8 = st.beta_columns((1,1.5))
    col8.write('----')

def parameter_experiment(state):
    st.title("parameter_experiment")


def hough_lines(state):
    st.title("hough_lines")


def extrapolate_lines(state):
    st.title("extrapolate_lines")


def stabilize_lines(state):
    st.title("stabilize_lines")


def fill_lines(state):
    st.title("fill_lines")



def display_state_values(state):
    #dict_filters = {"Industry" : state.add_selectbox_industry,
    #                "Sub Industry" : state.add_multibox_sub_industry,
    #                "Revenue model" : state.add_selectbox_revenue_model,
    #                "Country" : state.add_selectbox_country,
    #                "Business type" : state.add_selectbox_business_type,
    #                "Company size" : state.add_selectbox_company_size,
    #                "Strategy type" : state.add_selectbox_strategy_type,
    #                "Objective" : state.add_selectbox_objective
    #                }
    #   st.write(dict_filters)

    df_filters = pd.DataFrame()
    df_filters['Filter'] = ["Industry", 
                            "Sub Industry", 
                            "Revenue model",
                            "Country" ,
                            "Business type",
                            "Company size",
                            "Strategy type",
                            "Objective"]
    df_filters['Value'] = [state.add_selectbox_industry,
                           state.add_multibox_sub_industry,
                           state.add_selectbox_revenue_model,
                           state.add_selectbox_country,
                           state.add_selectbox_business_type,
                           state.add_selectbox_company_size,
                           state.add_selectbox_strategy_type,
                           state.add_selectbox_objective]
    # st.write(df_filters)

    if st.button("Reset Search"):
        state.clear()



class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()