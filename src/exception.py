"""Compatibility exception for pipeline callers."""
import sys


class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        traceback = error_detail.exc_info()[2]
        detail = str(error_message)
        if traceback is not None:
            detail = f"{traceback.tb_frame.f_code.co_filename}:{traceback.tb_lineno}: {detail}"
        super().__init__(detail)
