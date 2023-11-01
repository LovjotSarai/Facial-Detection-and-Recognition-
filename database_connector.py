# database_connector.py

import mysql.connector

def create_db_connection(host_name, user_name, user_password, db_name):
    """Create a database connection."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except mysql.connector.Error as err:
        print(f"Error: '{err}'")

    return connection

def download_images(connection):
    """Download images from database."""
    cursor = connection.cursor()
    query = "SELECT image_data FROM images_table"
    cursor.execute(query)
    images = cursor.fetchall()

    for i, image_blob in enumerate(images):
        # Assuming the blob is in the format of a numpy array saved with cv2.imencode
        nparr = np.frombuffer(image_blob[0], np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save image
        cv2.imwrite(f'image_{i}.png', img_np)
    
    cursor.close()
    print("Images downloaded and saved locally.")
