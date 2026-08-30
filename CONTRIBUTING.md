# Guía de Contribución — MLOps_Secop

Este documento explica cómo trabajar en este repositorio: entorno, flujo de Git,
convención de commits y cómo abrir/mergear Pull Requests. Está pensado para
cualquier persona que se una al proyecto, incluido tu yo del futuro.

---

## 1. Configurar el entorno por primera vez

```powershell
git clone https://github.com/aemedina24/MLOps_Secop.git
cd MLOps_Secop
uv sync
uv run pre-commit install
```

**Validación:**

```powershell
uv run pytest
uv run ruff check .
```

Resultado esperado: `2 passed` y `All checks passed!`.

## 2. Si ya tenías el repo clonado y quieres los cambios más recientes

No necesitas volver a clonar. Basta con:

```powershell
git checkout main
git pull origin main
uv sync
```

`uv sync` es importante después de cada `pull`: si `pyproject.toml` o `uv.lock`
cambiaron, tu entorno local podría quedar desactualizado sin este paso.

---

## 3. Reglas del repositorio

- `main` está **protegida**: nadie puede hacer `push` directo ahí. Todo cambio
  entra por Pull Request.
- No se instalan librerías manualmente con `pip install`. Todo cambio de
  dependencias pasa por `pyproject.toml` + `uv add` / `uv sync`.
- No se sube código sin que pasen los hooks de `pre-commit` (Ruff).

---

## 4. Flujo de trabajo (branch → commit → PR → merge)

### Paso 1 — Actualiza `main` antes de crear tu rama

```powershell
git checkout main
git pull origin main
```

⚠️ Este paso se salta fácilmente. Si lo olvidas, tu rama nueva puede quedar
basada en una versión vieja de `main`.

### Paso 2 — Crea tu rama

Usa un prefijo que indique el tipo de cambio, igual que en los commits:

```powershell
git checkout -b <tipo>/<descripcion-corta>
```

Ejemplos: `feat/entrenamiento-modelo`, `fix/error-carga-datos`,
`test/validacion-features`, `chore/actualizar-dependencias`.

### Paso 3 — Trabaja y valida localmente

```powershell
uv run pytest
uv run ruff check .
```

No comitees si algo de esto falla.

### Paso 4 — Commit

```powershell
git add <archivo(s) específicos>
git commit -m "<tipo>: <descripción en minúsculas, sin punto final>"
```

⚠️ Evita `git add .` a menos que hayas revisado `git status` primero — es
fácil arrastrar cambios sueltos que no tienen relación con tu tarea.

### Paso 5 — Push

Primera vez en esta rama:

```powershell
git push -u origin <tipo>/<descripcion-corta>
```

Siguientes veces en la misma rama:

```powershell
git push
```

**Valida el resultado:** busca la línea `Total X (delta Y)`. Si dice
`Total 0`, probablemente olvidaste el `git commit` antes del push — revisa
con `git log --oneline -3` antes de seguir.

### Paso 6 — Abre el Pull Request

En GitHub, usa el link que te da la terminal después del push. Incluye en la
descripción: qué hace el cambio, por qué, y cómo probarlo.

### Paso 7 — Mergea con "Squash and merge"

⚠️ **Este proceso tiene DOS clics, no uno:**
1. Clic en el botón verde desplegable → elige "Squash and merge"
2. Se abre una ventana con el mensaje del commit final → clic en
   **"Confirm squash and merge"**

Si te quedas solo en el paso 1, el PR queda abierto sin mergear aunque
parezca que ya terminó.

Si ves un mensaje de "Sign up for free to join this conversation" en vez del
botón de merge, no estás autenticado en esa pestaña — inicia sesión y recarga.

Si ves "Review required" / "Merging is blocked": la protección de rama exige
aprobación. En equipos de una sola persona, ajusta esto en
**Settings → Branches → Edit regla de `main` → desmarca "Require approvals"**
(mantén marcado "Require a pull request before merging").

### Paso 8 — Sincroniza y limpia

```powershell
git checkout main
git pull origin main
git log --oneline -3        # confirma que tu commit aparece
uv run pytest               # valida en main
git branch -d <tu-rama>
git fetch --prune
git branch -a               # confirma que no quedaron ramas huérfanas
```

⚠️ Si `git branch -d` da error de "used by worktree", significa que sigues
parado en esa rama — haz `git checkout main` primero.

⚠️ Después de un *squash merge*, `git branch -d` puede mostrar un warning
("merged to origin/rama, but not yet merged to HEAD"). Es normal — el squash
crea un commit distinto al original, así que Git no lo reconoce por hash
exacto. Igual te deja borrar la rama.

---

## 5. Convención de Commits (Conventional Commits)

| Tipo | Cuándo usarlo | Ejemplo |
|---|---|---|
| `feat` | Agregas funcionalidad nueva | `feat: add data validation pipeline` |
| `fix` | Corriges un bug | `fix: resolve null pointer in preprocessing` |
| `docs` | Solo cambias documentación | `docs: update setup instructions` |
| `refactor` | Cambias código sin agregar funcionalidad ni corregir bugs | `refactor: simplify feature engineering logic` |
| `test` | Agregas o corriges tests | `test: add unit tests for model training` |
| `chore` | Tareas de mantenimiento (dependencias, CI, configs) | `chore: update pip dependencies` |
| `ci` | Cambios en CI/CD (GitHub Actions, pipelines) | `ci: add linting step to workflow` |
| `style` | Formato, espacios, punto y coma (sin cambio de lógica) | `style: fix indentation in train.py` |

---

## 6. Errores comunes (y cómo detectarlos)

Estos son tropiezos reales que ocurrieron construyendo este proyecto — se
documentan para que no se repitan sin darse cuenta.

**🔴 Push que no sube nada**
Síntoma: `git push` muestra `Total 0 (delta 0)`.
Causa: se hizo `push` sin haber hecho `commit` antes.
Cómo detectarlo: revisa `git log --oneline -3` — si tu cambio no aparece
como el commit más reciente, no se comiteó.

**🔴 PR que parece mergeado pero sigue "Open"**
Síntoma: el mensaje "No conflicts with base branch" aparece, pero al volver
a `main` no está el cambio.
Causa: solo se hizo clic en el primer botón de merge, sin confirmar en la
segunda ventana.
Cómo detectarlo: en la página del PR, el estado debe decir **"Merged"**
(morado), no **"Open"** (verde).

**🟡 Commit hecho en `main` en vez de en la rama de trabajo**
Síntoma: `git status` dice `On branch main` cuando se esperaba estar en una
rama de feature.
Causa: se olvidó ejecutar `git checkout -b <rama>` antes de editar archivos.
Solución: `git stash`, crear la rama correcta, `git stash pop`.

**🟡 `git branch -d` falla con "used by worktree"**
Causa: se intenta borrar la rama en la que se está parado actualmente.
Solución: `git checkout main` antes de borrar.

**🟡 Ramas remotas "fantasma" en `git branch -a`**
Síntoma: una rama ya borrada en GitHub sigue apareciendo localmente.
Causa: la información de ramas remotas queda en caché local.
Solución: `git fetch --prune`.

---

## 7. Estructura del proyecto

Ver `README.md` para la arquitectura completa y el roadmap por fases.
