import logging

# set config specify folder to save log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    filename="log.txt",
    filemode="captcha_log.txt",
)
