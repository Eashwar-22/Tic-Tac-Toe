# Tic Tac Toe

__Objective :__
* Replicate the game with two modes - Human vs Human & Human vs Computer.
* Human vs Computer is implemented using ANN based on the data generated by Human vs Human mode.
The idea is to mimic human gameplay without the need for algorithms like Minimax to predict the possible moves.

__Setback :__
* Generalisation of the deep learning model depends on how much the Human vs Human mode is being played as more data is needed from that. 
* Hence, the model takes time to generalise results - developing each of them by versions.

---
| Steps | Actions | Development |
| -|-|-|
| Game| Human vs Human mode is complete | Deployment in pygame module is needed | 
| Human Data | Generated on the go | Need atleast 1000 rows of data for first gen model to generalise |
| Training | |
| Implementing Human vs Computer | | 


---
__Sample Data__ (in one Human vs Human session) <br>
_Note : 1 session can contain multiple games._

<img width="1195" alt="Screenshot 2021-11-28 at 1 41 15 PM" src="https://user-images.githubusercontent.com/86509452/143734843-5db9b1b9-d559-485c-8c4c-6c7c05b54201.png">

__Description__ <br>
* player : X or O's turn
* 
