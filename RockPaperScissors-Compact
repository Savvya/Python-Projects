import random

choice = input(
    "Do you want to play a game of rock paper scissors with me?(yes or no)"
).lower()

while choice == 'yes':

    user_choice = input("Choose wisely (Rock, Paper or Scissors?) ")
    possibleChoices = ["Rock", "Paper", "Scissors"]
    computer_boi = random.choice(possibleChoices)

    if user_choice == "Rock" and computer_boi == "Rock" or user_choice == "Paper" and computer_boi == "Paper" or user_choice == "Scissors" and computer_boi == "Scissors":
        print("It's a TIE!")
    elif (user_choice == "Rock" and computer_boi == "Scissors") or (
            user_choice == "Paper" and computer_boi == "Rock"
    ) or user_choice == "Scissors" and computer_boi == "Paper":
        print("You won!")
    elif user_choice == "Paper" and computer_boi == "Scissors" or user_choice == "Scissors" and computer_boi == "Rock" or user_choice == "Rock" and computer_boi == "Paper":
        print("Computer wins!")
    else:
        print("This is not an Option. Try reading the choices")

    print(
        f"\nYou chose {user_choice} and the computer chose {computer_boi}.\n")

    choice = input("Do you want to continue?")
