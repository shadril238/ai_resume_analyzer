# **Starting a Django Project in Conda Environment**

## **1. Activate Your Conda Environment**
If you already have a Conda environment, activate it:
```bash
conda activate shadril238  # Replace with your actual environment name
```
If you haven’t created one, do:
```bash
conda create --name shadril238 python=3.10 -y
conda activate shadril238
```

---

## **2. Install Project Dependencies**
If your project has a `requirements.txt` file, install dependencies:
```bash
pip install -r requirements.txt
```
Otherwise, manually install Django and DRF:
```bash
pip install django djangorestframework
```

---

## **3. Set Up Environment Variables**
If your project uses **`.env` files**, install `python-dotenv`:
```bash
pip install python-dotenv
```
Then create a `.env` file if it doesn’t exist:
```bash
touch .env
```
Add necessary environment variables inside `.env`:
```
DJANGO_SECRET_KEY=your_secret_key
DEBUG=True
DATABASE_URL=your_database_url
```

---

## **4. Apply Migrations**
Run the following command to apply database migrations:
```bash
python manage.py migrate
```

---

## **5. Create a Superuser (Optional)**
If your project has **Django Admin**, create an admin user:
```bash
python manage.py createsuperuser
```
Follow the prompts to set up a username and password.

---

## **6. Run the Django Server**
Start the development server:
```bash
python manage.py runserver
```
By default, it runs at `http://127.0.0.1:8000/`.

---

## **7. Open the Project in Your Browser**
- API endpoints (if using DRF): `http://127.0.0.1:8000/api/`
- Django Admin: `http://127.0.0.1:8000/admin/`

---

## **8. (Optional) Debugging Issues**
- Check if the correct Conda environment is activated:
  ```bash
  conda info --envs
  ```
- If `manage.py` doesn’t run, check installed packages:
  ```bash
  pip list
  ```

Now your **Django Conda project is running successfully!** 🚀

