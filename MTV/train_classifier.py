from speech_act_classifier import train_and_save_classifier
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train the speech act classifier')
    parser.add_argument('--save_path', type=str, default='best_model.pth',
                      help='Path to save the trained model')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    args = parser.parse_args()

    print(f"Training classifier for {args.num_epochs} epochs...")
    print(f"Model will be saved to: {args.save_path}")
    
    train_and_save_classifier(
        save_path=args.save_path,
        num_epochs=args.num_epochs
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main() 