/**
 * The front page.
 *
 * The design's nameplate, alone on the paper, above a single rule and two
 * fields. It borrows the masthead's vocabulary rather than inventing a login
 * aesthetic — Instrument Serif for ARGUS, the double rule under it, Space Mono
 * for everything that is a fact rather than a sentence — because this is the
 * first screen a plant sees and it should already look like the product.
 *
 * Three behaviours that are not decoration:
 *
 * * **A freshly imaged appliance gets a different screen.** No accounts exist,
 *   so no password can work; showing a form would be a lie. It prints the one
 *   command that fixes it instead.
 * * **A rate-limited login says so, with the wait.** "Invalid username or
 *   password" repeated at an operator who is locked out is how support calls
 *   are generated.
 * * **The failure message is the API's, verbatim.** The API deliberately
 *   answers the same sentence for a wrong password, an unknown user and a
 *   disabled account, and the console must not helpfully distinguish them.
 */

import { useEffect, useRef, useState } from 'react';
import { ApiError } from '../api/client';
import { useSession } from '../state/Session';

export function Login() {
  const { signIn, notice, code } = useSession();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const first = useRef<HTMLInputElement>(null);

  useEffect(() => {
    first.current?.focus();
  }, []);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await signIn(username.trim(), password);
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.detail
          : 'The appliance did not answer. Check that `argus api` is running.',
      );
      setPassword('');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <div className="login__sheet rise">
        <div className="login__plate">
          <div className="login__title">ARGUS</div>
          <div className="login__strap">The Plant Observer</div>
        </div>

        {code === 'no_accounts' ? (
          <div className="login__bootstrap">
            <div className="eyebrow" style={{ marginBottom: 10 }}>No accounts on this appliance</div>
            <p className="login__prose">
              Argus will not answer a single request until somebody exists to make it. Create the
              first administrator on the box itself — physical access is the only credential a
              fresh appliance has, and that is deliberate.
            </p>
            <pre className="login__command">argus user add ops --role ADMIN</pre>
            <p className="login__prose">
              It prompts for the password, and will read one from stdin instead when there is no
              terminal, so an offline installer can do this unattended. Reload this page afterwards.
            </p>
          </div>
        ) : (
          <form className="login__form" onSubmit={submit}>
            {notice ? <div className="login__notice">{notice}</div> : null}

            <label className="login__field">
              <span className="eyebrow">Username</span>
              <input
                ref={first}
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                autoComplete="username"
                autoCapitalize="off"
                spellCheck={false}
                disabled={busy}
              />
            </label>

            <label className="login__field">
              <span className="eyebrow">Password</span>
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
                disabled={busy}
              />
            </label>

            {error ? <div className="login__error">{error}</div> : null}

            <button
              className="login__submit"
              type="submit"
              disabled={busy || !username.trim() || !password}
            >
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </form>
        )}

        <div className="login__foot">
          <span>ON-PREM · ALL PROCESSING ON THIS APPLIANCE</span>
          <span>NO RAW VIDEO LEAVES THE PLANT NETWORK</span>
        </div>
      </div>
    </div>
  );
}

/** Shown for the fraction of a second before `/api/auth/session` answers.
 *  A blank page here reads as a broken appliance; the nameplate reads as one
 *  that is starting. */
export function Splash() {
  return (
    <div className="login">
      <div className="login__sheet">
        <div className="login__plate">
          <div className="login__title">ARGUS</div>
          <div className="login__strap">The Plant Observer</div>
        </div>
      </div>
    </div>
  );
}
