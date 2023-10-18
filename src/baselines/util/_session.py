
import uuid
from baselines.constants import USER_SESSION_FOLDER


def get_new_session():
    session_path = f'{USER_SESSION_FOLDER}session_{str(uuid.uuid4())[:8]}'
    return session_path
