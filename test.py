import os
import requests
import logging
from supabase import create_client, Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    logger.error("SUPABASE_URL or SUPABASE_ANON_KEY environment variables are not set.")
    raise ValueError("Please set the SUPABASE_URL and SUPABASE_ANON_KEY environment variables.")
else:
    logger.debug(f"Supabase URL: {SUPABASE_URL}")
    logger.debug(f"Supabase Anon Key: {SUPABASE_ANON_KEY}")

# User credentials
USER_EMAIL = 'k@k.com'
USER_PASSWORD = 'test'
logger.debug(f"User Email: {USER_EMAIL}")

try:
    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    logger.info("Supabase client initialized successfully.")

    # Sign in the user
    auth_response = supabase.auth.sign_in_with_password({
        'email': USER_EMAIL,
        'password': USER_PASSWORD,
    })
    logger.debug(f"Auth Response: {auth_response}")

    # Check if authentication was successful
    if auth_response.user is None:
        logger.error("Authentication failed. Please check your email and password.")
        raise ValueError("Authentication failed. Please check your email and password.")
    else:
        logger.info("User authenticated successfully.")

    # Retrieve the user's ID
    user_id = auth_response.user.id
    logger.debug(f"User ID: {user_id}")

    # Define the bucket and file path
    bucket_name = 'UserTFLite'
    file_path = f'{user_id}/gaze_model.tflite'
    logger.debug(f"Bucket Name: {bucket_name}")
    logger.debug(f"File Path: {file_path}")

    # Generate the public URL for the file
    public_url_response = supabase.storage.from_(bucket_name).get_public_url(file_path)
    public_url = public_url_response  # Since public_url_response is already a string
    logger.debug(f"Public URL: {public_url}")

    # Download the file
    response = requests.get(public_url)
    logger.debug(f"HTTP GET Response: {response.status_code}")

    if response.status_code == 200:
        # Save the file locally
        with open('gaze_model.tflite', 'wb') as file:
            file.write(response.content)
        logger.info('File downloaded and saved successfully.')
    else:
        logger.error(f'Failed to download file. HTTP status code: {response.status_code}')
except Exception as e:
    logger.exception("An error occurred during the process.")
