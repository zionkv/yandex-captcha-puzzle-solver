
# YandexCaptchaPuzzleSolver

YandexCaptchaPuzzleSolver is a service to bypass Yandex Captcha (Puzzle).

## How it works

YandexCaptchaPuzzleSolver starts a server, that can solve yandex captcha of puzzle type :

![Yandex puzzle captcha view](https://github.com/user-attachments/assets/ed71b6b1-5260-43dc-ba3b-40bea1826aa5)


and it waits for user requests.
For get some site valid token (result of solving), need to send request to docker (see Installation):

    curl -XPOST 'http://localhost:20081/get_token' \
      -H 'Content-Type: application/json' \
      --data-raw '{"maxTimeout": 120000, "url": "SITE FOR SOLVE", "yandex_key": "YANDEX KEY"}'

YANDEX KEY you can get from source code of target page, usualy it starts with **ysc1_** string.

Response example:

    {"status":"ok","message":"Challenge solved!","startTimestamp":1733819749.824522,"endTimestamp":1733819774.119855,"solution":{"status":"ok","url":"<MY SITE>","cookies":[{"name":"receive-cookie-deprecation","value":"1","domain":".yandex.ru","path":"/","secure":true},{"name":"session-cookie","value":"180fc3e2fb41df94e50241d9d00b084574552116189d7515109f2424d43b405a76cd9ae4255944b2d868fe358dc27d53","domain":".some.domain","path":"/","secure":false}],"user_agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36","token":"dD0xNzMzODE5NzY3O2k9MjE3LjY1LjIuMjI5O0Q9NzAzQzI4OTlFRDBFQTBFRTM1ODE3MUFBMzRFMkFDRURDQkQzQTlFMDgwMzM4QjMzRDJEODlDMTczMTEyQTk5ODZDODkyMEQxNzA4QTBFN0I4MTkxQzVCRkQ3RjRDMzExQ0E3Qjg1NkRDOEM4MDZENTFEM0JERENFODUzNzlEMTYzODY2MkM5RDg2RjIwQUEwNzc7dT0xNzMzODE5NzY3NTk4OTEyNjU3O2g9ZjI3ZWY0OWUxZmUyN2EzNWQ4OTNmM2IzYzM5YTQwNWU="}}

## Installation

It is recommended to install using a Docker container because the project depends on an external browser that is
already included within the image.

We provide a `docker-compose.yml` configuration file.
Clone this repository and execute `docker compose up -d` to start the container.



