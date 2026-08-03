create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text,
  free_credits_remaining integer not null default 5 check (free_credits_remaining >= 0),
  subscription_status text not null default 'free',
  stripe_customer_id text unique,
  stripe_subscription_id text,
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles alter column free_credits_remaining set default 5;

alter table public.profiles enable row level security;

do $$
begin
  if not exists (
    select 1
    from pg_policies
    where schemaname = 'public'
      and tablename = 'profiles'
      and policyname = 'Users can read their own profile'
  ) then
    create policy "Users can read their own profile"
    on public.profiles
    for select
    to authenticated
    using (auth.uid() = user_id);
  end if;
end;
$$;

create or replace function public.consume_free_credit(target_user_id uuid)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  did_consume boolean;
begin
  update public.profiles
  set
    free_credits_remaining = free_credits_remaining - 1,
    updated_at = now()
  where
    user_id = target_user_id
    and free_credits_remaining > 0
    and coalesce(subscription_status, 'free') not in ('active', 'trialing')
  returning true into did_consume;

  return coalesce(did_consume, false);
end;
$$;

create table if not exists public.credit_purchases (
  stripe_checkout_session_id text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  credits_added integer not null check (credits_added > 0),
  created_at timestamptz not null default now()
);

alter table public.credit_purchases enable row level security;

create or replace function public.grant_image_credit_purchase(
  target_user_id uuid,
  checkout_session_id text,
  credits_to_add integer
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  inserted_count integer;
begin
  if credits_to_add <= 0 or nullif(checkout_session_id, '') is null then
    return false;
  end if;

  insert into public.credit_purchases (stripe_checkout_session_id, user_id, credits_added)
  values (checkout_session_id, target_user_id, credits_to_add)
  on conflict (stripe_checkout_session_id) do nothing;

  get diagnostics inserted_count = row_count;
  if inserted_count = 0 then
    return false;
  end if;

  update public.profiles
  set free_credits_remaining = free_credits_remaining + credits_to_add, updated_at = now()
  where user_id = target_user_id;

  if not found then
    raise exception 'Profile not found for user %', target_user_id;
  end if;
  return true;
end;
$$;

revoke all on function public.grant_image_credit_purchase(uuid, text, integer) from public, anon, authenticated;
grant execute on function public.grant_image_credit_purchase(uuid, text, integer) to service_role;
