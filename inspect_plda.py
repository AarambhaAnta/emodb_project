try:
    from speechbrain.processing.PLDA_LDA import PLDA
    import torch
    plda = PLDA()
    print("Attributes of PLDA:", dir(plda))
except ImportError:
    print("SpeechBrain not installed")
except Exception as e:
    print(f"Error: {e}")