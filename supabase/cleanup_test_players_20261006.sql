-- Manual cleanup only. Exact test identities from historical reports / generated cleanup.sql.
-- Not executed: no database management permission. Never delete by nickname/IP.
begin;
delete from qqt_private.players where player_id in (
  '1cbd8dd8-1980-4b3f-9892-83164f27d5a1'::uuid,
  '3c162283-4dc5-4c30-b23c-83079e8d9e27'::uuid,
  '4465444b-09ea-4043-ac85-d159f45ec4fb'::uuid,
  '45ab592f-7202-4b88-9818-585c1f1a8929'::uuid,
  '6a0231d5-5905-4125-a291-11f36dcb83e1'::uuid,
  'ad141562-447e-4562-9175-bf5b53256d75'::uuid,
  'e48b877c-15f8-4b84-ba43-a71162add0dd'::uuid
);
commit;
