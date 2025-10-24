# My Attempt at Developing a GPT Model from Scratch

This is my attempt at developing a GPT model from scratch with my understanding.

---

## Try it Yourself

1. **Set up your own virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate     # On macOS/Linux
   venv\Scripts\activate        # On Windows
   
2. **Install the requirements**  
   ```bash
   pip install -r requirements.txt

3. **Train the model** 

   After training the model, the script generates 500 new tokens and prints the generated text.
    ```bash
   python gpt.py

4. **View the results**
   
   The results for my run are saved in the `results.txt` file.
   
   All the hyperparameters are defined in `gpt.py`.

## My Results

- **Training environment:** Google Colab (Python 3 Google Compute Engine backend — GPU)
- **GPU:** 1× T4
- **Training time:** ~50 minutes