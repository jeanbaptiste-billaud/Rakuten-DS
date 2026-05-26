from fastapi import FastAPI, Header, HTTPException, Response
from auth_utils import check_token

app = FastAPI(title="Auth Service")


@app.get("/check")
def check_auth(token: str = Header(...)):
    try:
        role = check_token(token)
        response = Response(status_code=200)
        response.headers["X-User-Role"] = role
        return response
    except Exception:
        raise HTTPException(status_code=403)

