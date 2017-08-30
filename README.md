# MNIST-DigitsRecognition

Repository contains simple solution for handwritten digits recognition. 
Solution is build using Python3 and Keras library. 

## Preliminaries

Projects uses just two non-standard libraries Keras and numpy. Correct versions 
of these libraries can be easily installed from project directory using this command:

```bash
pip3 install -r requirements.txt
```

## Running the solution

Solution for the digit recognition can be found in `eval_ensemble.py` file. File can be run using this commad:

```python
python3 eval_ensemble.py
```

Script creates ensemble from 5 independently trained CNN models and test the 
resulting ensemble on MNIST test data. Measured error rate is printed to stdout. 

Similar result with additional comments can be seen in Jupyter notebook 
`MNIST-DigitsRecognition.ipynb`.
