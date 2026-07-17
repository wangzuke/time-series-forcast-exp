# Setup GitHub SSH (免密推送配置)

为当前项目配置独立的 SSH key 实现免密码 GitHub 推送，适用于共享账号（如 root）场景，不影响其他用户。

## 使用场景

- 共享服务器上需要隔离 Git 身份
- 新项目需要配置免密 GitHub 推送
- 用户说"配置 GitHub SSH"、"免密推送"、"setup github ssh"

## 步骤

执行以下操作，每一步确认成功后再继续：

### 1. 收集信息

向用户确认：
- GitHub 用户名
- 关联邮箱
- SSH key 名称（默认 `id_ed25519_<用户名>`）

### 2. 检查现有配置

```bash
git remote -v
ls -la ~/.ssh/
git config user.name; git config user.email
```

### 3. 生成专属 SSH key

```bash
ssh-keygen -t ed25519 -C "<邮箱>" -f ~/.ssh/id_ed25519_<用户名> -N ""
```

如果 key 已存在，询问用户是否复用。

### 4. 输出公钥，指导用户添加到 GitHub

```bash
cat ~/.ssh/id_ed25519_<用户名>.pub
```

告知用户：
1. 打开 https://github.com/settings/keys
2. 点击 **New SSH key**
3. 粘贴公钥并保存

等待用户确认已添加后继续。

### 5. 配置项目级 Git

```bash
# 设置项目级用户信息
git config user.name "<用户名>"
git config user.email "<邮箱>"

# 设置项目级 SSH 命令
git config core.sshCommand "ssh -i ~/.ssh/id_ed25519_<用户名> -o IdentitiesOnly=yes"

# 切换 remote 为 SSH 协议（如果当前是 HTTPS）
git remote set-url origin git@github.com:<用户名>/<仓库名>.git
```

### 6. 验证连接

```bash
ssh -T -i ~/.ssh/id_ed25519_<用户名> -o IdentitiesOnly=yes git@github.com 2>&1 || true
git fetch origin
```

确认输出包含 `Hi <用户名>! You've successfully authenticated`。

### 7. 完成总结

告知用户：
- SSH key 路径
- 所有配置都在项目级别（`.git/config`），不影响全局
- 后续直接 `git push` / `git pull` 即可免密操作
