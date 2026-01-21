# Milimo Music

A music generation application powered by HeartMuLa models.

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

## Setup Instructions

### 1. HeartLib & Model Weights

**Crucial Step**: You must download the large model weights manually as they are excluded from the repository.

1.  Navigate to the `heartlib` directory:
    ```bash
    cd heartlib
    ```

2.  Install the library in editable mode:
    ```bash
    pip install -e .
    ```

3.  **Download Pretrained Models**:
    You need to download the checkpoints into the `heartlib/ckpt` directory. You can use Hugging Face or ModelScope.

    **Using Hugging Face CLI:**
    ```bash
    # Install hf-hub if not present: pip install huggingface_hub[cli]

    hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
    hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B'
    hf download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss'
    ```

    **Directory Structure Verification**:
    After downloading, ensure your `heartlib/ckpt` folder looks like this:
    ```
    heartlib/ckpt/
    ├── HeartCodec-oss/
    ├── HeartMuLa-oss-3B/
    ├── gen_config.json
    └── tokenizer.json
    ```

### 2. Backend

1.  Navigate to the `backend` directory:
    ```bash
    cd ../backend
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the server:
    ```bash
    python -m app.main
    ```
    The backend will start at `http://localhost:8000`.

### 3. Frontend

1.  Navigate to the `frontend` directory:
    ```bash
    cd ../frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Start the development server:
    ```bash
    npm run dev
    ```
    The frontend will be available at `http://localhost:5173`.

## Author

**Mainza Kangombe**  
[LinkedIn Profile](https://www.linkedin.com/in/mainza-kangombe-6214295)
