import datetime
import inspect
from abc import ABC, abstractmethod
from typing import List

import yagmail
from tgsend import ParseMode, Telegram


START_ICON = "🏃‍♀️"
ERROR_ICON = "❌"
END_ICON = "🏁"


class Notifier(ABC):
    def __init__(self, title=None):
        self.title = title
        self.start_time = None
        self.end_time = None

    @abstractmethod
    def notify_start(self, message=None, **kwargs):
        pass

    @abstractmethod
    def notify_error(self, error_message):
        pass

    @abstractmethod
    def notify_end(self, message=None, **kwargs):
        pass

    def _format_kwargs(self, kwargs):
        return "\n".join([f"{k}:  {v}" for k, v in kwargs.items()])

    def _get_run_time(self):
        if self.start_time and self.end_time:
            elapsed = self.end_time - self.start_time
            elapsed -= datetime.timedelta(microseconds=elapsed.microseconds)
            return f"\n\nStarted: {self.start_time:%Y-%m-%d %H:%M:%S}\nEnded: {self.end_time:%Y-%m-%d %H:%M:%S}\nRun time: {elapsed}"
        elif self.start_time:
            return f"\n\nStarted: {self.start_time:%Y-%m-%d %H:%M:%S}"
        else:
            return ""


class TelegramNotifier(Notifier):
    name = "telegram"

    def __init__(self, recipients: List[str] = None, title=None):
        super().__init__(title=title)
        self.recipients = recipients
        self.telegram = Telegram(parse_mode=ParseMode.HTML)

    def notify_start(self, message=None, **kwargs):
        self.start_time = datetime.datetime.now()
        title = " Started: " + self.title
        text = self.telegram.format_fixed(self._format_kwargs(kwargs))
        if message:
            text = message + "\n" + text
        text += self._get_run_time()
        if self.recipients:
            for recipient in self.recipients:
                self.telegram.send_message(text, title=title, chat_id=recipient, icon=START_ICON)
        else:
            self.telegram.send_message(text, title=title, icon=START_ICON)

    def notify_error(self, error_message):
        self.end_time = datetime.datetime.now()
        title = "FAILED: " + self.title
        text = error_message + self._get_run_time()
        if self.recipients:
            for recipient in self.recipients:
                self.telegram.send_message(text, title=title, chat_id=recipient, icon=ERROR_ICON)
        else:
            self.telegram.send_message(text, title=title, icon=ERROR_ICON)

    def notify_end(self, message=None, **kwargs):
        self.end_time = datetime.datetime.now()
        title = "Finished: " + self.title
        text = self.telegram.format_fixed(self._format_kwargs(kwargs))
        if message:
            text = message + "\n" + text
        text += self._get_run_time()
        if self.recipients:
            for recipient in self.recipients:
                self.telegram.send_message(text, title=title, chat_id=recipient, icon=END_ICON)
        else:
            self.telegram.send_message(text, title=title, icon=END_ICON)


class EmailNotifier(Notifier):
    name = "email"

    def __init__(self, recipients: List[str], sender: str = None, title=None):
        super().__init__(title=title)
        self.recipients = recipients
        self.mail_sender = yagmail.SMTP(user=sender)

    def notify_start(self, message=None, **kwargs):
        self.start_time = datetime.datetime.now()
        title = START_ICON + " Started: " + self.title
        text = self._format_kwargs(kwargs)
        if message:
            text = message + "\n" + text
        text += self._get_run_time()
        for recipient in self.recipients:
            self.mail_sender.send(to=recipient, subject=title, contents=text)

    def notify_error(self, error_message):
        self.end_time = datetime.datetime.now()
        title = ERROR_ICON + " FAILED: " + self.title
        for recipient in self.recipients:
            self.mail_sender.send(to=recipient, subject=title, contents=error_message + self._get_run_time())

    def notify_end(self, message=None, **kwargs):
        self.end_time = datetime.datetime.now()
        title = END_ICON + " Finished: " + self.title
        text = self._format_kwargs(kwargs)
        if message:
            text = message + "\n" + text
        text += self._get_run_time()
        for recipient in self.recipients:
            self.mail_sender.send(to=recipient, subject=title, contents=text)


NOTIFIER_CLASSES = {}
for name, obj in globals().copy().items():
    if inspect.isclass(obj) and issubclass(obj, Notifier) and obj != Notifier:
        NOTIFIER_CLASSES[obj.name] = obj
