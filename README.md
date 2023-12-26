# captcha_recognision

the project for the dm course in Radboud University

current results for the cnn model:
loss: 0.1666 - accuracy: 0.9447 - val_loss: 0.5296 - val_accuracy: 0.8703

             precision    recall  f1-score   support

           2       0.97      0.88      0.93        43
           3       0.93      1.00      0.96        40
           4       0.93      0.90      0.91        48
           5       0.95      0.90      0.92        40
           6       0.90      0.95      0.92        38
           7       0.92      0.92      0.92        39
           8       0.92      1.00      0.96        44
           b       0.94      0.89      0.92        37
           c       0.88      0.90      0.89        49
           d       0.93      0.93      0.93        40
           e       0.83      0.81      0.82        42
           f       0.86      0.86      0.86        42
           g       0.91      0.98      0.94        43
           m       0.58      0.76      0.66        29
           n       0.88      0.76      0.82        85
           p       0.97      0.94      0.95        32
           w       0.93      0.85      0.89        33
           x       0.79      0.90      0.84        42
           y       0.94      0.89      0.92        37

    accuracy                           0.89       803
   macro avg       0.89      0.90      0.89       803
weighted avg       0.90      0.89      0.89       803

accuracy: 89%



current results for the knn:

              precision    recall  f1-score   support

           2       0.97      0.27      0.42       140
           3       0.84      0.97      0.90        37
           4       0.83      0.95      0.88        40
           5       0.92      0.92      0.92        38
           6       0.75      1.00      0.86        30
           7       0.87      0.94      0.91        36
           8       0.77      1.00      0.87        37
           b       0.80      0.93      0.86        30
           c       0.86      0.90      0.88        48
           d       0.68      0.93      0.78        29
           e       0.76      0.94      0.84        33
           f       0.83      0.88      0.85        40
           g       0.76      0.97      0.85        36
           m       0.55      0.88      0.68        24
           n       0.86      0.77      0.82        83
           p       0.84      1.00      0.91        26
           w       0.90      0.90      0.90        30
           x       0.71      0.92      0.80        37
           y       0.77      0.93      0.84        29

    accuracy                           0.80       803
   macro avg       0.80      0.89      0.83       803
weighted avg       0.84      0.80      0.78       803

accuracy: 80%