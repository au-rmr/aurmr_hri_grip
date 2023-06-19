# Supported Primitives

## Gripping Primitives

<table>
	<tr><th>Primitive Name</th><th>Parameters</th><th>Description</th></tr>
	<tr>
		<td>front_approach</td>
		<td>
			<table>
				<tr><td>x</td><td>float</td><td>x position relative to camera frame</td></tr>
				<tr><td>y</td><td>float</td><td>y position relative to camera frame</td></tr>
				<tr><td>z</td><td>float</td><td>z position relative to closest point of target object</td></tr>
			</table>
		</td>
		<td>Move the end effector to the given position</td>
	</tr>
	<tr>
		<td>set_gripper_width</td>
		<td>
			<table>
				<tr><td>width</td><td>float</td><td>width of the gripper in mm</td></tr>
			</table>
		</td>
		<td>Adjust the width of the gripper</td>
	</tr>
	<tr>
		<td>set_gripper_rot</td>
		<td>
			<table>
				<tr><td>angle</td><td>float</td><td>angle in degrees</td></tr>
			</table>
		</td>
		<td>Adjust the gripper rotation along the Z-axis</td>
	</tr>
</table>

## Probing Primitives

<table>
	<tr><th>Primitive Name</th><th>Parameters</th><th>Description</th></tr>
	<tr>
		<td>probe_gripper</td>
		<td>
			<table>
				<tr><td>magnitude</td><td>float</td><td>magnitude of the probing motion (between 0-1)</td></tr>
			</table>
		</td>
		<td>Close then reopen gripper slightly</td>
	</tr>
	<tr>
		<td>probe_lift</td>
		<td>
			<table>
				<tr><td>magnitude</td><td>float</td><td>magnitude of the probing motion (between 0-1)</td></tr>
			</table>
		</td>
		<td>Up down movement</td>
	</tr>
	<tr>
		<td>probe_pull</td>
		<td>
			<table>
				<tr><td>magnitude</td><td>float</td><td>magnitude of the probing motion (between 0-1)</td></tr>
			</table>
		</td>
		<td>Back and forth movement</td>
	</tr>
	<tr>
		<td>probe_push</td>
		<td>
			<table>
				<tr><td>magnitude</td><td>float</td><td>magnitude of the probing motion (between 0-1)</td></tr>
			</table>
		</td>
		<td>Forward then back movement</td>
	</tr>
	<tr>
		<td>probe_sides</td>
		<td>
			<table>
				<tr><td>magnitude</td><td>float</td><td>magnitude of the probing motion (between 0-1)</td></tr>
			</table>
		</td>
		<td>Side-to-side movement</td>
	</tr>
</table>

## Extraction Primitives


<table>
	<tr><th>Primitive Name</th><th>Parameters</th><th>Description</th></tr>
	<tr>
		<td>lift</td>
		<td>
			<table>
				<tr><td>distance</td><td>float</td><td>distance to lift in mm</td></tr>
			</table>
		</td>
		<td>Lift the gripped object</td>
	</tr>
	<tr>
		<td>pull_back</td>
		<td>
			<table>
			</table>
		</td>
		<td>Pull the gripped object out of the pod along the z-axis</td>
	</tr>
