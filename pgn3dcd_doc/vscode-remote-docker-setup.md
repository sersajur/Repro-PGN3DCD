# Налаштування: VSCode + хмарна VM + Docker Compose

Мета: редагувати й запускати код прямо в контейнері на VM, ніби локально.

---

## 1. Локально (на своїй машині)

**Постав VSCode**, потім розширення:
- **Remote Development** (пак: одразу Remote-SSH + Dev Containers)

**Налаштуй SSH** — у `~/.ssh/config`:

```
Host my-vm
    HostName 12.34.56.78
    User ubuntu
    IdentityFile ~/.ssh/your-key.pem
    ForwardAgent yes
```

**Додай GitHub-ключ у локальний ssh-agent** (щоб не класти секрети на VM):

```bash
ssh-add ~/.ssh/your-github-key
```

Перевір доступ: `ssh my-vm` має пускати без пароля.

---

## 2. На VM (один раз перевір, що є)

- встановлені **Docker** і **docker compose**
- твій користувач у групі `docker`:

```bash
sudo usermod -aG docker $USER
# після цього перелогінитись (вийти й зайти по ssh)
```

---

## 3. Заходимо на VM з VSCode

`Cmd/Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → `my-vm`

VSCode поставить свій сервер на VM. Далі все (файли, термінал, дебаг) — на VM.

---

## 4. Стягуємо код і піднімаємо стек

У терміналі VSCode (він уже на VM):

```bash
git clone git@github.com:you/your-repo.git
cd your-repo
docker compose up -d
```

> Agent forwarding робить так, що `git clone`/`push` працюють твоїми локальними ключами — на VM секретів не лишається.

**Перевір, що код — це volume, а не запечений в образ** (інакше зміни зникатимуть). У `docker-compose.yml`:

```yaml
services:
  app:
    build: .
    volumes:
      - ./src:/app/src        # код монтується всередину
    command: uvicorn main:app --reload --host 0.0.0.0
```

---

## 5. Заходимо всередину контейнера

`Cmd/Ctrl+Shift+P` → **Dev Containers: Attach to Running Container** → обери свій python-сервіс.

Відкриється нове вікно «всередині» контейнера.

> **⚠️ Якщо побачиш `Missing GLIBC >= 2.28`** — образ надто старий (напр. Ubuntu 18.04), VS Code Server туди не стане.
> Це не лагодиться налаштуванням. Тоді пропускай кроки 5–6 і працюй так:
> - **редагуєш** — у вікні **Remote-SSH** (папка репо на VM; код і так змонтований у контейнер через `volumes`)
> - **запускаєш** — у терміналі: `docker compose exec <сервіс> bash`
> - **дебажиш** — через `debugpy`, див. розділ 7

---

## 6. Донаштовуємо контейнер і працюємо

У вікні контейнера:
- відкрий робочу папку (напр. `/app`)
- постав розширення **Python** (Microsoft), за потреби **Pylance**
  *(вони ставляться саме в контекст контейнера — це нормально)*

Готово. Редагуєш файли → сервіс сам перезавантажується (`--reload`) → дебаг і breakpoints працюють.

---

## 7. Дебаг через debugpy (коли attach недоступний)

Дає breakpoints у контейнері **без** VS Code Server усередині — тому вимоги до glibc не діють.
Ідея: процес у контейнері відкриває порт і чекає, VSCode (з боку VM) чіпляється до нього.

### Що поставити

Всередині контейнера:

```bash
pip install debugpy
```

Краще одразу в `Dockerfile` (безпечно, чистий python-пакет):

```dockerfile
RUN pip3 install debugpy && rm -rf /root/.cache
```

### Конфіг у VSCode

У вікні **Remote-SSH** (не контейнера), у репо створи `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to container",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        { "localRoot": "${workspaceFolder}", "remoteRoot": "/tp3d" }
      ]
    }
  ]
}
```

> `remoteRoot` — шлях коду **всередині** контейнера. Без правильного `pathMappings` breakpoints не спрацюють.
> При `network_mode: host` порт видно на VM напряму — прокидувати нічого не треба.

### Як запускати

**1.** У терміналі контейнера — замість звичайного `python3` :

```bash
docker compose exec <сервіс> bash

python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    trainSiamKPConv.py output_dir=/output/test
```

Процес завмре й чекатиме IDE.

**2.** У VSCode тисни **F5** (конфіг «Attach to container») — скрипт побіжить далі й зупиниться на breakpoint.

Далі все як завжди: змінні, стек, step over/in, Debug Console.

### Що відрізняється від звичайного дебагу

- запуск **у два кроки** (термінал → F5), а не просто F5
- після завершення скрипта сесія вмирає — для повтору знову запускаєш команду
- автодоповнення по бібліотеках контейнера (torch тощо) не працює — Pylance їх не бачить з VM

---

## Щоденний цикл після налаштування

### Варіант А — attach працює

1. Remote-SSH: Connect to Host → `my-vm`
2. (за потреби) `docker compose up -d`
3. Dev Containers: Attach to Running Container
4. редагуєш, запускаєш, комітиш — усе у вікні контейнера

### Варіант Б — attach недоступний (старий glibc)

Редагуєш на VM, запускаєш у контейнері. Код той самий — він змонтований через `volumes`.

**На початку дня:**

1. **Remote-SSH: Connect to Host** → твоя VM
2. Відкрий папку репо **на VM** (напр. `~/Repro-PGN3DCD`) — це єдине робоче вікно
3. Підніми shell-контейнер, якщо не живий:
   ```bash
   docker compose up -d gcp-gpu-shell
   docker compose ps          # має бути Up
   ```

**Тримай два термінали у VSCode:**

| Термінал | Де | Для чого |
|----------|-----|----------|
| №1 | на VM | git, `docker compose ps`, `logs` |
| №2 | у контейнері (`docker compose exec gcp-gpu-shell bash`) | запуск скриптів |

**Робочий цикл:**

4. **Редагуєш** у вікні Remote-SSH — зміни одразу видно в контейнері (`${PROJECT_PATH:-.}:/tp3d`), синхронізувати нічого не треба
5. **Запускаєш** у терміналі №2 (він уже всередині, заходити щоразу не треба):
   ```bash
   python3 trainSiamKPConv.py output_dir=/output/test
   ```
6. **Дебажиш** — там само через `debugpy` + F5 (розділ 7)
7. **Комітиш** — у вікні Remote-SSH, звичайний Source Control

> Автодоповнення по `torch`/PyG не працюватиме (Pylance не бачить `/venv` контейнера) — на запуск і дебаг це не впливає.

---

## Шпаргалка

| Дія | Команда / крок |
|-----|----------------|
| Підключитись до VM | Remote-SSH: Connect to Host |
| Підняти стек | `docker compose up -d` |
| Зайти в контейнер (IDE) | Dev Containers: Attach to Running Container |
| Зайти в контейнер (термінал) | `docker compose exec <сервіс> bash` |
| Запустити скрипт під дебагом | `python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client script.py` |
| Під'єднати дебагер | **F5** у VSCode |
| Логи сервісу | `docker compose logs -f app` |
| Перезапустити сервіс | `docker compose restart app` |
| Зупинити стек | `docker compose down` |
