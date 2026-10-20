/**
 * Users & roles — the design's block, wired to the appliance.
 *
 * It used to be an honest empty state saying there was no authentication. There
 * is now, so this is the roster: the design's circular initials, the name, the
 * role as a Space Mono chip, and — the part the design could only imply — the
 * controls that change them.
 *
 * ADMIN only. Rendered at all only for an ADMIN session, and refused by the API
 * for everyone else regardless; a roster is a list of who works in the control
 * room and what each of them can reach.
 *
 * Two rules are shown rather than merely enforced:
 *
 * * The **last enabled ADMIN** cannot be disabled or demoted. The control is
 *   disabled with the reason on hover, instead of being offered and then
 *   answering 409 — an appliance that has locked its owner out needs a person
 *   with shell access to recover.
 * * Every destructive action **ends that account's sessions**, and the panel
 *   says so, because "disabled" that leaves an open browser working until its
 *   cookie expires is not disabled.
 */

import { useState } from 'react';
import { ApiError, api } from '../api/client';
import type { Role, User, UserRoster } from '../api/types';
import { initials } from './Masthead';
import { ErrorBar } from './marks';
import { useResource } from '../lib/hooks';

export function UsersPanel({ minPasswordLength }: { minPasswordLength: number }) {
  const roster = useResource(() => api.users(), []);
  const [busy, setBusy] = useState<number | 'new' | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const run = async (who: number | 'new', action: () => Promise<unknown>) => {
    setBusy(who);
    setFailure(null);
    try {
      await action();
      roster.reload();
    } catch (cause) {
      setFailure(cause instanceof ApiError ? cause.detail : String(cause));
    } finally {
      setBusy(null);
    }
  };

  const data = roster.data;

  return (
    <div className="panel">
      <div style={{ fontSize: 'var(--fs-meta)', fontWeight: 600, marginBottom: 4 }}>Users &amp; roles</div>
      <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginBottom: 14 }}>
        Accounts on this appliance. Every verdict in Review Triage is recorded against the person
        who signed in, and a disabled account is signed out of every browser at once.
      </div>

      <ErrorBar error={roster.error} />
      {failure ? <div className="users__note users__note--bad">{failure}</div> : null}

      {data ? (
        <>
          <div>
            {data.users.map((user) => (
              <Row
                key={user.id}
                user={user}
                roster={data}
                busy={busy === user.id}
                onRole={(role) => run(user.id, () => api.updateUser(user.id, { role }))}
                onToggle={() =>
                  run(user.id, () => api.updateUser(user.id, { disabled: !user.disabled }))
                }
                onRevoke={() => run(user.id, () => api.revokeSessions(user.id))}
              />
            ))}
          </div>

          <AddUser
            roles={data.roles}
            descriptions={data.role_descriptions}
            minPasswordLength={minPasswordLength}
            busy={busy === 'new'}
            onCreate={(body) => run('new', () => api.createUser(body))}
          />
        </>
      ) : (
        <div className="users__note">Loading accounts…</div>
      )}
    </div>
  );
}

function Row({
  user,
  roster,
  busy,
  onRole,
  onToggle,
  onRevoke,
}: {
  user: User;
  roster: UserRoster;
  busy: boolean;
  onRole: (role: Role) => void;
  onToggle: () => void;
  onRevoke: () => void;
}) {
  // The guard the API enforces, shown here so the control is greyed rather than
  // offered-then-refused. Mirrors `_refuse_if_last_admin` in argus/accounts.py.
  const lastAdmin = user.role === 'ADMIN' && !user.disabled && roster.admin_count <= 1;
  const isSelf = user.username === roster.self;
  const lockReason = lastAdmin
    ? 'This is the only enabled administrator — create another one first, or nobody can manage this appliance.'
    : undefined;

  return (
    <div className={`users__row ${user.disabled ? 'users__row--off' : ''}`}>
      <span className="users__initials">{initials(user.display_name || user.username)}</span>
      <div className="users__who">
        <div className="users__name">
          {user.display_name}
          {isSelf ? <span style={{ color: 'var(--faint)' }}> · you</span> : null}
        </div>
        <div className="users__meta">
          <span className="mono">{user.username}</span>
          {user.disabled ? ' · disabled' : ''}
          {' · '}
          {user.last_login_at
            ? `last signed in ${new Date(user.last_login_at).toLocaleString()}`
            : 'never signed in'}
        </div>
      </div>

      <select
        className="users__role"
        value={user.role}
        disabled={busy || lastAdmin}
        title={lockReason ?? roster.role_descriptions[user.role]}
        onChange={(event) => onRole(event.target.value as Role)}
      >
        {roster.roles.map((role) => (
          <option key={role} value={role}>
            {role}
          </option>
        ))}
      </select>

      <div className="users__actions">
        <button
          className="chip-button"
          disabled={busy}
          onClick={onRevoke}
          title="Sign this account out of every browser without changing its password"
        >
          REVOKE
        </button>
        <button
          className="chip-button"
          disabled={busy || (lastAdmin && !user.disabled)}
          onClick={onToggle}
          title={lockReason ?? (user.disabled ? 'Let this account sign in again' : 'Sign out and refuse this account')}
        >
          {user.disabled ? 'ENABLE' : 'DISABLE'}
        </button>
      </div>
    </div>
  );
}

function AddUser({
  roles,
  descriptions,
  minPasswordLength,
  busy,
  onCreate,
}: {
  roles: Role[];
  descriptions: Record<Role, string>;
  minPasswordLength: number;
  busy: boolean;
  onCreate: (body: {
    username: string;
    password: string;
    role: Role;
    display_name?: string;
  }) => void;
}) {
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<Role>('REVIEW');

  const ready = username.trim().length > 0 && password.length >= minPasswordLength;

  return (
    <form
      className="users__add"
      onSubmit={(event) => {
        event.preventDefault();
        if (!ready || busy) return;
        onCreate({
          username: username.trim(),
          password,
          role,
          display_name: displayName.trim() || undefined,
        });
        setUsername('');
        setDisplayName('');
        setPassword('');
      }}
    >
      {/* Labels above the boxes, not placeholders inside them. A form labelled
          only by its placeholders goes blank the moment somebody starts
          typing, and the constraint that matters here — the minimum password
          length — was living in exactly that disappearing ink. */}
      <label>
        <span className="eyebrow">Username</span>
        <input
          value={username}
          autoCapitalize="off"
          spellCheck={false}
          onChange={(event) => setUsername(event.target.value)}
        />
      </label>
      <label>
        <span className="eyebrow">Display name · optional</span>
        <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} />
      </label>
      <label>
        <span className="eyebrow">Password · at least {minPasswordLength} characters</span>
        <input
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
      </label>
      <label>
        <span className="eyebrow">Role</span>
        <select value={role} onChange={(event) => setRole(event.target.value as Role)}>
          {roles.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
      <div className="users__add--wide users__note">{descriptions[role]}</div>
      <button className="chip-button users__add--wide" type="submit" disabled={!ready || busy}>
        {busy ? 'CREATING…' : 'ADD ACCOUNT'}
      </button>
    </form>
  );
}
