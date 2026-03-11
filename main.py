from trainer.train_model import train_model
from interact.Use_Model import use_model


if __name__ == "__main__":
    print("Apple Stock Price Prediction - Training")
    print("=" * 50)
    
    success = train_model()
    
    if success:
        print("\nTraining completed successfully!")
        print("You can now use Use_Model.py to make predictions.")
        
        use_model()
    else:
        print("\nTraining failed. Please check the error messages above.")
        sys.exit(1)
