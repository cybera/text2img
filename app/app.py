import streamlit as st
import requests
from PIL import Image
import plotly.express as px


def main():
    st.title("Text2Img: Stable Diffusion App")
    st.info(
        "Source: [HuggingFace Stable Diffusion with Diffusers blog](https://huggingface.co/blog/stable_diffusio)."
    )

    with st.form("my_form"):
        prompt = st.text_input(
            "Prompt: ",
            max_chars=100,
        )

        col1, col2 = st.columns(2)

        with col1:
            num_inference_steps = st.number_input(
                "No. of Inference Steps: ", value=50, max_value=500, min_value=5, step=1
            )

        with col2:
            guidance_scale = st.slider(
                label="Guidance scale", min_value=5.0, max_value=50.0, value=7.5
            )

        submitted = st.form_submit_button("Submit")

        if submitted:
            if not prompt == "":
                with st.spinner("Running inference..."):
                    response = requests.get(
                        url="http://fastapi:8000/text2img",
                        params={
                            "prompt": prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        },
                        stream=True,
                    ).raw

                    im = Image.open(response)

                    fig = px.imshow(im)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(
                    "Prompt cannot be empty. Please enter a valid prompt to start inferencing.."
                )


if __name__ == "__main__":
    main()
