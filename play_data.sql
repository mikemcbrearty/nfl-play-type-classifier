select
  cast(down as float) / 4,
  cast(ydstogo as float) / 10,
  cast(iif(trim(substr(yrdln, 1,3))=posteam, 100-trim(substr(yrdln,4)), trim(substr(yrdln,4))) as float) / 100 as ydstogo_100,
  cast(score_differential as float) / 60,
  iif(game_half='Half1', 0.0, 1.0),
  cast(half_seconds_remaining as float) / 1800,
  cast(posteam_timeouts_remaining as float) / 3,
  cast(defteam_timeouts_remaining as float) / 3,
  case play_type
    when 'pass' then 0
    when 'run' then 1
    when 'punt' then 2
    when 'field_goal' then 3
  end
from nfl
where play_type in ('pass','run','punt','field_goal')
  and game_half<>'Overtime'
  and down<>'NA'
;
